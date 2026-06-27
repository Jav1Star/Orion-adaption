#!/usr/bin/env python3
import json
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from collections import deque
from pathlib import Path
from tqdm import tqdm


POLL_INTERVAL_SECONDS = 5.0
PORT_STEP = 150
TERMINATE_GRACE_SECONDS = 30.0
CRASH_MARKERS_SUBSTR = (
    "Watchdog exception",
    "Engine crash handling finished; re-raising signal 11",
    "Stopping the route, the agent has crashed",
)


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

    if len(model_gpus) == 1 and len(gpu_ranks) > 1:
        model_gpus = model_gpus * len(gpu_ranks)
    if len(carla_gpus) == 1 and len(gpu_ranks) > 1:
        carla_gpus = carla_gpus * len(gpu_ranks)

    if not (len(gpu_ranks) == len(model_gpus) == len(carla_gpus)):
        raise ValueError(
            "GPU_RANK_LIST, MODEL_GPU, and CARLA_GPU must have the same number of entries"
        )

    return gpu_ranks, model_gpus, carla_gpus


def load_route_ids(routes_file):
    tree = ET.parse(routes_file)
    return [route.attrib["id"] for route in tree.iter("route")]


def is_failed_status(status):
    return isinstance(status, str) and status.strip().startswith("Failed")


def should_skip_route(result_file, resume_enabled):
    if not resume_enabled or not result_file.exists():
        return False
    return not is_failed_status(get_route_status(result_file))


def get_route_status(result_file):
    if not result_file.exists():
        return None
    try:
        with open(result_file, "r", encoding="utf-8") as infile:
            data = json.load(infile)
    except Exception:
        return None

    checkpoint = data.get("_checkpoint", {})
    global_record = checkpoint.get("global_record", {})
    status = global_record.get("status")
    if isinstance(status, str):
        return status.strip()

    records = checkpoint.get("records", [])
    if records:
        record_status = records[0].get("status")
        if isinstance(record_status, str):
            return record_status.strip()
    return None


def cleanup_worker_ports(port, tm_port):
    for current_port in (port, port + 1, port + 2, tm_port):
        subprocess.run(
            ["fuser", "-k", "-9", f"{current_port}/tcp"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )


def build_worker_specs(base_port, base_tm_port, gpu_ranks, model_gpus, carla_gpus):
    workers = []
    for index, (gpu_rank, model_gpu, carla_gpu) in enumerate(
        zip(gpu_ranks, model_gpus, carla_gpus)
    ):
        workers.append(
            {
                "worker_id": index,
                "model_gpu": model_gpu,
                "carla_gpu": carla_gpu,
                "port": base_port + index * PORT_STEP,
                "tm_port": base_tm_port + index * PORT_STEP,
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
    checkpoint_endpoint = Path(os.environ["CHECKPOINT_ENDPOINT"]).resolve()
    save_path = Path(os.environ["SAVE_PATH"]).resolve()
    leaderboard_root = Path(os.environ["LEADERBOARD_ROOT"]).resolve()
    team_agent = os.environ["TEAM_AGENT"]
    team_config = os.environ["TEAM_CONFIG"]
    repetitions = os.environ.get("REPETITIONS", "1")
    debug_challenge = os.environ.get("DEBUG_CHALLENGE", "0")
    record_path = os.environ.get("RECORD_PATH", "")
    challenge_track = os.environ.get("CHALLENGE_TRACK_CODENAME", "SENSORS")
    base_port = int(os.environ.get("PORT", "30000"))
    base_tm_port = int(os.environ.get("TM_PORT", "50000"))
    resume_enabled = parse_bool_env("RESUME", default=True)
    visualization_enabled = parse_bool_env("ORION_EVAL_VISUALIZATION", default=False)

    route_root = checkpoint_endpoint.parent / "routes"
    result_dir = route_root / "res"
    out_dir = route_root / "out"
    err_dir = route_root / "err"
    directories = [result_dir, out_dir, err_dir]
    if visualization_enabled:
        directories.append(save_path)
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

    gpu_ranks, model_gpus, carla_gpus = resolve_gpu_lists()
    workers = build_worker_specs(base_port, base_tm_port, gpu_ranks, model_gpus, carla_gpus)

    route_ids = load_route_ids(routes_file)
    pending_routes = deque(route_ids)
    running_workers = []
    route_result_files = []
    progress_bar = tqdm(total=len(route_ids), desc="B2D routes", dynamic_ncols=True)

    log_message(
        progress_bar,
        f"Starting Orion Bench2Drive route scheduler: routes={len(route_ids)} workers={len(workers)}",
    )
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
            route_save_path = None
            if visualization_enabled:
                route_save_path = save_path / route_id
                route_save_path.mkdir(parents=True, exist_ok=True)

            if should_skip_route(result_file, resume_enabled):
                route_result_files.append(str(result_file))
                route_status = get_route_status(result_file)
                if route_status is not None:
                    log_message(
                        progress_bar,
                        f"[skip] route={route_id} status={route_status} result={result_file}",
                    )
                else:
                    log_message(progress_bar, f"[skip] route={route_id} result={result_file}")
                progress_bar.update(1)
                refresh_progress(
                    progress_bar,
                    running_count=len(running_workers),
                    pending_count=len(pending_routes),
                )
                loop_progress = True
                continue

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
            if route_save_path is not None:
                env["SAVE_PATH"] = str(route_save_path)
            else:
                env.pop("SAVE_PATH", None)

            stdout_handle = open(out_file, "w", encoding="utf-8")
            stderr_handle = open(err_file, "w", encoding="utf-8")
            stdout_handle.write(" ".join(command) + "\n")
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
                "out_handle": stdout_handle,
                "err_handle": stderr_handle,
                "port": worker["port"],
                "tm_port": worker["tm_port"],
            }
            running_workers.append(worker)
            route_result_files.append(str(result_file))
            log_message(
                progress_bar,
                f"[start] worker={worker['worker_id']} route={route_id} "
                f"model_gpu={worker['model_gpu']} carla_gpu={worker['carla_gpu']} "
                f"port={worker['port']} tm_port={worker['tm_port']}",
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
            cleanup_worker_ports(job["port"], job["tm_port"])
            log_message(
                progress_bar,
                f"[done] worker={worker['worker_id']} route={job['route_id']} return_code={return_code}",
            )
            route_status = get_route_status(Path(job["result_file"]))
            if route_status is not None and route_status not in {"Completed", "Perfect"}:
                log_message(
                    progress_bar,
                    f"[failed] route={job['route_id']} status={route_status}",
                )
            elif return_code != 0:
                log_message(
                    progress_bar,
                    f"[failed] route={job['route_id']} return_code={return_code}",
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
    subprocess.run(merge_command, cwd=str(leaderboard_root.parent), check=True)
    progress_bar.close()


if __name__ == "__main__":
    main()
