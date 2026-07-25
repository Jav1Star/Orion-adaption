#!/usr/bin/env python3
import atexit
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from collections import deque
from pathlib import Path
from tqdm import tqdm


POLL_INTERVAL_SECONDS = 5.0
PORT_STEP = 150
CARLA_RPC_PORT_COUNT = 3
TERMINATE_GRACE_SECONDS = 30.0
CARLA_TERMINATE_GRACE_SECONDS = 5.0
CRASH_MARKERS_SUBSTR = (
    "Watchdog exception",
    "Engine crash handling finished; re-raising signal 11",
    "Stopping the route, the agent has crashed",
)
SUCCESS_STATUSES = {"Completed", "Perfect"}


def parse_bool_env(name, default=False):
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    return str(raw_value).strip().lower() in {"1", "true", "yes", "on"}


def parse_int_list(raw_value):
    values = []
    for item in str(raw_value).split(","):
        item = item.strip()
        if not item:
            continue
        values.append(int(item))
    return values


def broadcast_gpu_list(name, values, target_length):
    if len(values) == target_length:
        return values
    if len(values) == 1:
        return values * target_length
    raise ValueError(
        f"{name} must have either 1 entry or {target_length} entries, got {len(values)}"
    )


def resolve_gpu_rank_mapping(values, target_length):
    if len(values) >= target_length:
        return values[:target_length]
    return broadcast_gpu_list("GPU_RANK_LIST", values, target_length)


def resolve_gpu_lists():
    gpu_rank_list_raw = os.environ.get("GPU_RANK_LIST", "")
    if gpu_rank_list_raw.strip():
        gpu_ranks = parse_int_list(gpu_rank_list_raw)
    else:
        gpu_ranks = [int(os.environ.get("GPU_RANK", "0"))]

    model_gpu_raw = os.environ.get("MODEL_GPU", "")
    carla_gpu_raw = os.environ.get("CARLA_GPU", "")

    model_gpus = parse_int_list(model_gpu_raw) if model_gpu_raw.strip() else list(gpu_ranks)
    carla_gpus = parse_int_list(carla_gpu_raw) if carla_gpu_raw.strip() else list(gpu_ranks)

    # worker 数量由模型和 CARLA 的实际分配决定；GPU_RANK_LIST 只补齐 legacy gpu-rank 映射。
    worker_count = max(len(model_gpus), len(carla_gpus))
    gpu_ranks = resolve_gpu_rank_mapping(gpu_ranks, worker_count)
    model_gpus = broadcast_gpu_list("MODEL_GPU", model_gpus, worker_count)
    carla_gpus = broadcast_gpu_list("CARLA_GPU", carla_gpus, worker_count)

    return gpu_ranks, model_gpus, carla_gpus


def load_route_ids(routes_file):
    tree = ET.parse(routes_file)
    return [route.attrib["id"] for route in tree.iter("route")]


def resolve_route_ids(routes_file):
    route_ids = load_route_ids(routes_file)
    route_ids_raw = os.environ.get("ORION_EVAL_ROUTE_IDS", "").strip()
    if not route_ids_raw:
        return route_ids

    selected_route_ids = [item.strip() for item in route_ids_raw.split(",") if item.strip()]
    route_id_set = set(route_ids)
    missing_route_ids = [route_id for route_id in selected_route_ids if route_id not in route_id_set]
    if missing_route_ids:
        raise ValueError(f"ORION_EVAL_ROUTE_IDS contains routes not in {routes_file}: {missing_route_ids}")
    return selected_route_ids


def is_failed_status(status):
    return isinstance(status, str) and status.strip().startswith("Failed")


def is_success_status(status):
    return isinstance(status, str) and status.strip() in SUCCESS_STATUSES


def get_route_status(result_file):
    if not result_file.exists():
        return None
    try:
        with open(result_file, "r", encoding="utf-8") as infile:
            data = json.load(infile)
    except Exception:
        return None

    checkpoint = data.get("_checkpoint", {})
    # 单条 route 结果里，更具体的失败原因通常记录在 records[0].status。
    records = checkpoint.get("records", [])
    if records:
        record_status = records[0].get("status")
        if isinstance(record_status, str):
            return record_status.strip()

    global_record = checkpoint.get("global_record", {})
    status = global_record.get("status")
    if isinstance(status, str):
        return status.strip()
    return None


def should_queue_route(result_file):
    if not result_file.exists():
        return True
    route_status = get_route_status(result_file)
    return route_status is None or is_failed_status(route_status)


def cleanup_route_cache(route_id, result_file, out_file, err_file, route_save_path, progress_bar):
    # failed route 必须删掉旧 checkpoint，否则 leaderboard 的 resume 会直接跳过执行。
    removed_paths = []
    for cache_path in (result_file, out_file, err_file):
        if cache_path.exists():
            cache_path.unlink()
            removed_paths.append(str(cache_path))
    if route_save_path.exists():
        if route_save_path.is_dir():
            shutil.rmtree(route_save_path)
        else:
            route_save_path.unlink()
        removed_paths.append(str(route_save_path))
    if removed_paths:
        log_message(
            progress_bar,
            f"[cleanup] route={route_id} removed={len(removed_paths)}",
        )


def cleanup_worker_ports(ports_to_cleanup):
    for current_port in ports_to_cleanup:
        for protocol in ("tcp", "udp"):
            subprocess.run(
                ["fuser", "-k", "-9", f"{current_port}/{protocol}"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )


def read_carla_pid_file(pid_file):
    if not pid_file.exists():
        return []
    carla_pids = []
    with open(pid_file, "r", encoding="utf-8") as infile:
        for line in infile:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            label, raw_pid = parts
            try:
                pid = int(raw_pid)
            except ValueError:
                continue
            if pid > 1:
                carla_pids.append((label, pid))
    return carla_pids


def process_group_exists(pid):
    try:
        os.killpg(pid, 0)
    except ProcessLookupError:
        return False
    return True


def process_exists(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def send_process_group_signal(pid, signal_value):
    try:
        os.killpg(pid, signal_value)
    except ProcessLookupError:
        return False
    return True


def cleanup_carla_processes(job, progress_bar):
    # evaluator 会把本 route 启动的 CARLA 进程组写入 pid 文件，这里只清理自己的组。
    carla_pids = read_carla_pid_file(Path(job["carla_pid_file"]))
    live_pids = []
    for label, pid in carla_pids:
        if not process_group_exists(pid):
            continue
        if not send_process_group_signal(pid, signal.SIGTERM):
            continue
        live_pids.append((label, pid))
        log_message(
            progress_bar,
            f"[carla_cleanup] route={job['route_id']} label={label} pgid={pid} signal=TERM",
        )

    if live_pids:
        deadline = time.time() + CARLA_TERMINATE_GRACE_SECONDS
        while time.time() < deadline and any(process_group_exists(pid) for _, pid in live_pids):
            time.sleep(0.2)

    for label, pid in live_pids:
        if not process_group_exists(pid):
            continue
        if send_process_group_signal(pid, signal.SIGKILL):
            log_message(
                progress_bar,
                f"[carla_cleanup] route={job['route_id']} label={label} pgid={pid} signal=KILL",
            )


def cleanup_stale_carla_pid_files(pid_dir, progress_bar):
    # 上次 Ctrl-C 可能留下 CARLA 进程组；重启同一实验前先按 pid 文件回收。
    for pid_file in sorted(pid_dir.glob("*.pid")):
        job = {
            "route_id": pid_file.stem,
            "carla_pid_file": str(pid_file),
        }
        cleanup_carla_processes(job, progress_bar)
        pid_file.unlink(missing_ok=True)


def cleanup_stale_evaluators(outdir_root, progress_bar):
    # 旧 evaluator 可能仍占 traffic-manager 端口，只回收当前实验输出目录对应的进程。
    checkpoint_prefix = str(outdir_root / "routes" / "res") + os.sep
    ps_output = subprocess.run(
        ["ps", "-eo", "pid=,cmd="],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    ).stdout
    stale_pids = []
    for line in ps_output.splitlines():
        stripped_line = line.strip()
        if not stripped_line:
            continue
        raw_pid, _, command = stripped_line.partition(" ")
        if "leaderboard_evaluator.py" not in command:
            continue
        if f"--checkpoint={checkpoint_prefix}" not in command:
            continue
        try:
            pid = int(raw_pid)
        except ValueError:
            continue
        if pid == os.getpid():
            continue
        stale_pids.append(pid)

    for pid in stale_pids:
        try:
            os.kill(pid, signal.SIGTERM)
            log_message(progress_bar, f"[stale_eval_cleanup] pid={pid} signal=TERM")
        except ProcessLookupError:
            pass

    if stale_pids:
        deadline = time.time() + CARLA_TERMINATE_GRACE_SECONDS
        while time.time() < deadline and any(process_exists(pid) for pid in stale_pids):
            time.sleep(0.2)

    for pid in stale_pids:
        if not process_exists(pid):
            continue
        try:
            os.kill(pid, signal.SIGKILL)
            log_message(progress_bar, f"[stale_eval_cleanup] pid={pid} signal=KILL")
        except ProcessLookupError:
            pass


def close_job_handles(job):
    for handle_key in ("out_handle", "err_handle"):
        handle = job.get(handle_key)
        if handle is None or handle.closed:
            continue
        handle.flush()
        handle.close()


def cleanup_running_jobs(workers, progress_bar):
    for worker in workers:
        job = worker.get("job")
        if job is None:
            continue
        process = job["process"]
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=TERMINATE_GRACE_SECONDS)
            except subprocess.TimeoutExpired:
                process.kill()
        close_job_handles(job)
        cleanup_carla_processes(job, progress_bar)
        cleanup_worker_ports(job["ports_to_cleanup"])
        worker["job"] = None


def is_socket_port_available(port, socket_type):
    with socket.socket(socket.AF_INET, socket_type) as sock:
        try:
            sock.bind(("0.0.0.0", port))
        except OSError:
            return False
    return True


def is_port_available(port):
    return is_socket_port_available(port, socket.SOCK_STREAM) and is_socket_port_available(
        port, socket.SOCK_DGRAM
    )


def get_carla_required_ports(port, tm_port):
    # CARLA 启动会占用 RPC 相邻端口；分配和清理必须按同一组端口处理。
    return {port, port + 1, port + 2, tm_port}


def find_free_worker_ports(preferred_port, preferred_tm_port, reserved_ports):
    offset = 0
    while True:
        port = preferred_port + offset
        tm_port = preferred_tm_port + offset
        required_ports = get_carla_required_ports(port, tm_port)
        if required_ports.isdisjoint(reserved_ports) and all(
            is_port_available(current_port) for current_port in required_ports
        ):
            reserved_ports.update(required_ports)
            return port, tm_port, required_ports
        offset += CARLA_RPC_PORT_COUNT


def build_worker_specs(base_port, base_tm_port, gpu_ranks, model_gpus, carla_gpus):
    workers = []
    reserved_ports = set()
    for index, (gpu_rank, model_gpu, carla_gpu) in enumerate(
        zip(gpu_ranks, model_gpus, carla_gpus)
    ):
        preferred_port = base_port + index * PORT_STEP
        preferred_tm_port = base_tm_port + index * PORT_STEP
        port, tm_port, ports_to_cleanup = find_free_worker_ports(
            preferred_port, preferred_tm_port, reserved_ports
        )
        workers.append(
            {
                "worker_id": index,
                "gpu_rank": gpu_rank,
                "model_gpu": model_gpu,
                "carla_gpu": carla_gpu,
                "preferred_port": preferred_port,
                "preferred_tm_port": preferred_tm_port,
                "port": port,
                "tm_port": tm_port,
                "ports_to_cleanup": sorted(ports_to_cleanup),
                "job": None,
            }
        )
    return workers


def log_message(progress_bar, message):
    if progress_bar is None:
        print(message, flush=True)
    else:
        progress_bar.write(message)


def refresh_progress(progress_bar, running_count, pending_count):
    if progress_bar is None:
        return
    progress_bar.set_postfix(running=running_count, pending=pending_count)
    progress_bar.refresh()


def read_log_lines(job):
    lines = []
    for handle_key, path_key in (("out_handle", "out_file"), ("err_handle", "err_file")):
        handle = job.get(handle_key)
        if handle and not handle.closed:
            handle.flush()
        log_path = job.get(path_key)
        if not log_path or not os.path.exists(log_path):
            continue
        try:
            with open(log_path, "r", encoding="utf-8") as infile:
                lines.extend(infile.readlines())
        except Exception:
            continue
    return lines


def has_crash_marker(lines):
    for line in lines:
        for marker in CRASH_MARKERS_SUBSTR:
            if marker in line:
                return True
    return False


def extract_failure_hint(lines):
    interesting_markers = (
        "Traceback",
        "RuntimeError:",
        "FileNotFoundError:",
        "ModuleNotFoundError:",
        "CUDA out of memory",
        "Killed",
        "Stopping the route",
        "Watchdog exception",
        "Engine crash handling finished",
    )
    for line in reversed(lines):
        clean_line = line.strip()
        if not clean_line:
            continue
        if any(marker in clean_line for marker in interesting_markers):
            return clean_line
    return None


def describe_non_success(job, return_code):
    result_file = Path(job["result_file"])
    route_status = get_route_status(result_file)
    if return_code != 0:
        reason = f"return_code={return_code}"
        if route_status is not None:
            reason += f" status={route_status}"
        return reason
    if not result_file.exists():
        return "missing_result"
    if route_status is None:
        return "missing_status"
    if not is_success_status(route_status):
        return f"status={route_status}"
    return None


def log_non_success(progress_bar, route_id, reason, out_file=None, err_file=None, hint=None):
    message = f"[FAIL] route={route_id} reason={reason}"
    if out_file is not None:
        message += f" out={out_file}"
    if err_file is not None:
        message += f" err={err_file}"
    log_message(progress_bar, message)
    if hint:
        log_message(progress_bar, f"[FAIL_HINT] route={route_id} {hint}")


def log_failure_summary(progress_bar, failed_routes, checkpoint_endpoint):
    if not failed_routes:
        return
    log_message(
        progress_bar,
        f"[FAIL_SUMMARY] non_success_routes={len(failed_routes)} output={checkpoint_endpoint}",
    )
    for failure in failed_routes[:20]:
        log_message(
            progress_bar,
            f"[FAIL_SUMMARY] route={failure['route_id']} reason={failure['reason']} "
            f"out={failure['out_file']} err={failure['err_file']}",
        )
    if len(failed_routes) > 20:
        log_message(progress_bar, f"[FAIL_SUMMARY] ... {len(failed_routes) - 20} more")


def check_and_kill_dead_job(job, progress_bar):
    process = job["process"]
    if process.poll() is not None:
        return

    termination_requested_at = job.get("termination_requested_at")
    if termination_requested_at is not None:
        if time.time() - termination_requested_at >= TERMINATE_GRACE_SECONDS:
            log_message(
                progress_bar,
                f"[kill] route={job['route_id']} pid={process.pid} did not exit after terminate, sending kill",
            )
            process.kill()
        return

    lines = read_log_lines(job)
    if not lines:
        return
    if has_crash_marker(lines):
        log_message(
            progress_bar,
            f"[terminate] route={job['route_id']} pid={process.pid} crash marker detected in logs",
        )
        process.terminate()
        job["termination_requested_at"] = time.time()


def main():
    routes_file = os.environ["ROUTES"]
    outdir_root = Path(os.environ["OUTDIR_ROOT"]).resolve()
    # 所有闭环产物统一落在 outdir_root 下，便于做多次实验隔离。
    checkpoint_endpoint = Path(
        os.environ.get("CHECKPOINT_ENDPOINT", str(outdir_root / "orion_eval.json"))
    ).expanduser().resolve()
    save_path = Path(os.environ.get("SAVE_PATH", str(outdir_root / "records"))).expanduser().resolve()
    leaderboard_root = Path(os.environ["LEADERBOARD_ROOT"]).resolve()
    team_agent = os.environ["TEAM_AGENT"]
    team_config = os.environ["TEAM_CONFIG"]
    repetitions = os.environ.get("REPETITIONS", "1")
    debug_challenge = os.environ.get("DEBUG_CHALLENGE", "0")
    record_path = os.environ.get("RECORD_PATH", "")
    challenge_track = os.environ.get("CHALLENGE_TRACK_CODENAME", "SENSORS")
    base_port = int(os.environ.get("PORT", "30000"))
    base_tm_port = int(os.environ.get("TM_PORT", "50000"))
    visualization_enabled = parse_bool_env("ORION_EVAL_VISUALIZATION", default=False)
    max_route_attempts = max(1, int(os.environ.get("ORION_EVAL_MAX_ROUTE_ATTEMPTS", "2")))
    stage1_inference_budget = os.environ.get("ORION_STAGE1_INFERENCE_BUDGET", "")
    selected_route_ids = os.environ.get("ORION_EVAL_ROUTE_IDS", "").strip()

    route_root = outdir_root / "routes"
    result_dir = route_root / "res"
    out_dir = route_root / "out"
    err_dir = route_root / "err"
    pid_dir = route_root / "pids"
    directories = [outdir_root, result_dir, out_dir, err_dir, pid_dir, save_path]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

    cleanup_stale_evaluators(outdir_root, None)
    cleanup_stale_carla_pid_files(pid_dir, None)

    gpu_ranks, model_gpus, carla_gpus = resolve_gpu_lists()
    workers = build_worker_specs(base_port, base_tm_port, gpu_ranks, model_gpus, carla_gpus)
    atexit.register(cleanup_running_jobs, workers, None)

    route_ids = resolve_route_ids(routes_file)
    pending_routes = deque(route_ids)
    running_workers = []
    route_result_files = []
    failed_routes = []
    route_attempts = {route_id: 0 for route_id in route_ids}
    progress_bar = tqdm(total=len(route_ids), desc="B2D routes", dynamic_ncols=True)

    log_message(
        progress_bar,
        f"Starting Orion Bench2Drive route scheduler: routes={len(route_ids)} workers={len(workers)}",
    )
    log_message(
        progress_bar,
        f"GPU mapping: model={model_gpus} carla={carla_gpus}",
    )
    log_message(
        progress_bar,
        f"Route retry policy: max_route_attempts={max_route_attempts}",
    )
    log_message(
        progress_bar,
        f"Stage1 inference budget: {stage1_inference_budget or 'unset'}",
    )
    if selected_route_ids:
        log_message(progress_bar, f"Selected route ids: {selected_route_ids}")
    refresh_progress(progress_bar, running_count=0, pending_count=len(pending_routes))

    while pending_routes or running_workers:
        loop_progress = False
        for worker in workers:
            if worker["job"] is not None:
                continue
            if not pending_routes:
                continue

            route_id = pending_routes.popleft()
            result_file = result_dir / f"{route_id}.json"
            out_file = out_dir / f"{route_id}.log"
            err_file = err_dir / f"{route_id}.log"
            carla_pid_file = pid_dir / f"{route_id}.pid"
            route_save_path = save_path / route_id

            if not should_queue_route(result_file):
                route_result_files.append(str(result_file))
                route_status = get_route_status(result_file)
                if is_success_status(route_status):
                    log_message(
                        progress_bar,
                        f"[skip] route={route_id} status={route_status}",
                    )
                else:
                    reason = f"existing status={route_status or 'unknown'}"
                    log_non_success(
                        progress_bar,
                        route_id=route_id,
                        reason=reason,
                        out_file=out_file,
                        err_file=err_file,
                    )
                    failed_routes.append(
                        {
                            "route_id": route_id,
                            "reason": reason,
                            "out_file": str(out_file),
                            "err_file": str(err_file),
                        }
                    )
                progress_bar.update(1)
                refresh_progress(
                    progress_bar,
                    running_count=len(running_workers),
                    pending_count=len(pending_routes),
                )
                loop_progress = True
                continue
            route_status = get_route_status(result_file)
            route_attempts[route_id] += 1
            cleanup_route_cache(
                route_id,
                result_file,
                out_file,
                err_file,
                route_save_path,
                progress_bar,
            )
            if visualization_enabled:
                route_save_path.mkdir(parents=True, exist_ok=True)
            if carla_pid_file.exists():
                carla_pid_file.unlink()
            log_message(
                progress_bar,
                f"[queue] route={route_id} status={route_status or 'missing'} "
                f"attempt={route_attempts[route_id]}/{max_route_attempts}",
            )

            command = [
                sys.executable,
                str(leaderboard_root / "leaderboard" / "leaderboard_evaluator.py"),
                f"--routes={routes_file}",
                f"--routes-subset={route_id}",
                f"--repetitions={repetitions}",
                f"--track={challenge_track}",
                f"--checkpoint={result_file}",
                f"--agent={team_agent}",
                f"--agent-config={team_config}",
                f"--debug={debug_challenge}",
                f"--record={record_path}",
                f"--port={worker['port']}",
                f"--traffic-manager-port={worker['tm_port']}",
                f"--gpu-rank={worker['carla_gpu']}",
            ]

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(worker["model_gpu"])
            env["ORION_CARLA_PID_FILE"] = str(carla_pid_file)
            if route_save_path is not None:
                env["SAVE_PATH"] = str(route_save_path)
            else:
                env.pop("SAVE_PATH", None)

            stdout_handle = open(out_file, "w", encoding="utf-8")
            stderr_handle = open(err_file, "w", encoding="utf-8")
            stdout_handle.write(" ".join(command) + "\n")
            stdout_handle.write(
                # 关键调用点：CUDA_VISIBLE_DEVICES 会把物理模型卡重映射成进程内 cuda:0，OOM 日志里的 GPU 0 不是 CARLA 卡。
                f"[runner_env] CUDA_VISIBLE_DEVICES={worker['model_gpu']} "
                f"model_gpu_physical={worker['model_gpu']} "
                f"carla_gpu_physical={worker['carla_gpu']} "
                f"legacy_gpu_rank={worker['gpu_rank']}\n"
            )
            stdout_handle.flush()

            process = subprocess.Popen(
                command,
                cwd=str(leaderboard_root.parent),
                env=env,
                stdout=stdout_handle,
                stderr=stderr_handle,
            )

            worker["job"] = {
                "route_id": route_id,
                "process": process,
                "result_file": str(result_file),
                "out_file": str(out_file),
                "err_file": str(err_file),
                "carla_pid_file": str(carla_pid_file),
                "out_handle": stdout_handle,
                "err_handle": stderr_handle,
                "port": worker["port"],
                "tm_port": worker["tm_port"],
                "ports_to_cleanup": worker["ports_to_cleanup"],
            }
            running_workers.append(worker)
            route_result_files.append(str(result_file))
            log_message(
                progress_bar,
                f"[start] worker={worker['worker_id']} route={route_id} "
                f"model_gpu={worker['model_gpu']} carla_gpu={worker['carla_gpu']} "
                f"port={worker['port']} tm_port={worker['tm_port']} "
                f"preferred_port={worker['preferred_port']} "
                f"preferred_tm_port={worker['preferred_tm_port']}",
            )
            refresh_progress(
                progress_bar,
                running_count=len(running_workers),
                pending_count=len(pending_routes),
            )
            loop_progress = True

        active_workers = []
        for worker in running_workers:
            job = worker["job"]
            process = job["process"]
            return_code = process.poll()
            if return_code is None:
                check_and_kill_dead_job(job, progress_bar)
                active_workers.append(worker)
                continue

            job["out_handle"].flush()
            job["err_handle"].flush()
            job["out_handle"].close()
            job["err_handle"].close()
            cleanup_carla_processes(job, progress_bar)
            cleanup_worker_ports(job["ports_to_cleanup"])
            log_message(
                progress_bar,
                f"[done] worker={worker['worker_id']} route={job['route_id']} return_code={return_code}",
            )
            failure_reason = describe_non_success(job, return_code)
            if failure_reason is not None:
                hint = extract_failure_hint(read_log_lines(job))
                log_non_success(
                    progress_bar,
                    route_id=job["route_id"],
                    reason=failure_reason,
                    out_file=job["out_file"],
                    err_file=job["err_file"],
                    hint=hint,
                )
                if route_attempts[job["route_id"]] < max_route_attempts:
                    pending_routes.append(job["route_id"])
                    progress_bar.total += 1
                    log_message(
                        progress_bar,
                        f"[requeue] route={job['route_id']} "
                        f"next_attempt={route_attempts[job['route_id']] + 1}/{max_route_attempts}",
                    )
                else:
                    failed_routes.append(
                        {
                            "route_id": job["route_id"],
                            "reason": failure_reason,
                            "out_file": job["out_file"],
                            "err_file": job["err_file"],
                            "hint": hint,
                        }
                    )
            worker["job"] = None
            progress_bar.update(1)
            loop_progress = True

        running_workers = active_workers
        refresh_progress(
            progress_bar,
            running_count=len(running_workers),
            pending_count=len(pending_routes),
        )
        if (pending_routes or running_workers) and not loop_progress:
            time.sleep(POLL_INTERVAL_SECONDS)

    merge_inputs = []
    seen_files = set()
    for result_file in route_result_files:
        if result_file in seen_files:
            continue
        if not os.path.exists(result_file):
            continue
        seen_files.add(result_file)
        merge_inputs.append(result_file)

    merge_command = [
        sys.executable,
        str(leaderboard_root / "scripts" / "merge_statistics.py"),
        "-e",
        str(checkpoint_endpoint),
        "-f",
        *merge_inputs,
    ]
    log_message(
        progress_bar,
        f"Merging {len(merge_inputs)} route result files into {checkpoint_endpoint}",
    )
    log_failure_summary(progress_bar, failed_routes, checkpoint_endpoint)
    subprocess.run(merge_command, cwd=str(leaderboard_root.parent), check=True)
    progress_bar.close()


if __name__ == "__main__":
    main()
