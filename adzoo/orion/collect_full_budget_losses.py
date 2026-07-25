import argparse
import copy
import json
import os
import re
import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set

import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mmcv.datasets import build_dataset
from mmcv.models import build_model
from mmcv.parallel import collate
from mmcv.utils import Config, load_checkpoint, set_random_seed


@dataclass(frozen=True)
class SampleRecord:
    index: int
    measurement_path: str
    route_key: str
    route_id: str
    frame_id: int
    route_frame_index: int
    route_frame_count: int


class DirectIndexDataset(Dataset):
    """按外部 route scheduler 给出的 index 取样，避免 Dataset 自己随机换样。"""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        example = self.dataset.prepare_train_data(index)
        if example is None:
            return {
                "_collect_skip": True,
                "_collect_index": int(index),
                "_collect_reason": "prepare_train_data returned None",
            }
        return index, example


class RouteBatchSampler(Sampler[List[int]]):
    """按 route slot 组织 batch，让 DataLoader worker 提前准备样本。"""

    def __init__(self, records: Sequence[SampleRecord], route_batch_size: int):
        self.batches = [
            [record.index for record in batch_records]
            for batch_records in build_route_batches(records, route_batch_size)
        ]

    def __iter__(self) -> Iterator[List[int]]:
        return iter(self.batches)

    def __len__(self) -> int:
        return len(self.batches)


def parse_args():
    parser = argparse.ArgumentParser(description="Collect Orion full-budget per-sample driving losses.")
    parser.add_argument("config", help="current Orion adaption training config")
    parser.add_argument("checkpoint", help="stage1 final checkpoint")
    parser.add_argument("output_jsonl", help="output path, must end with _all.jsonl")
    parser.add_argument("route_batch_size", type=int, help="number of route slots per batch")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="DataLoader workers, default cfg.data.workers_per_gpu",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=2,
        help="DataLoader prefetch_factor when workers > 0",
    )
    parser.add_argument("--resume", action="store_true", help="append missing rows from an existing output_jsonl")
    parser.add_argument(
        "--done-jsonl",
        action="append",
        default=[],
        help="existing _all.jsonl to use as finished rows without writing to it; can be passed multiple times",
    )
    parser.add_argument("--num-route-shards", type=int, default=1, help="split routes into this many deterministic shards")
    parser.add_argument("--route-shard-id", type=int, default=0, help="current route shard id, in [0, num_route_shards)")
    return parser.parse_args()


def to_repo_relative(path: str) -> str:
    path_obj = Path(path)
    if not path_obj.is_absolute():
        return os.path.normpath(str(path_obj))
    try:
        return os.path.normpath(str(path_obj.resolve().relative_to(REPO_ROOT)))
    except ValueError:
        return os.path.normpath(str(path_obj))


def route_id_from_route_key(route_key: str) -> str:
    return Path(route_key).name


def disable_train_randomness(train_data_cfg):
    train_data_cfg = copy.deepcopy(train_data_cfg)
    # 关键调用点：reference 需要覆盖每个当前帧，空 3D GT 不能沿用训练过滤直接丢样本。
    train_data_cfg.filter_empty_gt = False
    train_data_cfg.full_budget_losses_jsonl = None
    pipeline = []
    for transform in train_data_cfg.pipeline:
        transform = copy.deepcopy(transform)
        if transform["type"] == "PhotoMetricDistortionMultiViewImage":
            continue
        if transform["type"] == "ResizeCropFlipRotImage":
            transform["training"] = False
        pipeline.append(transform)
    train_data_cfg.pipeline = pipeline
    return train_data_cfg


def _use_col_loss_from_config(config: Config) -> Optional[bool]:
    if "use_col_loss" in config:
        return bool(config["use_col_loss"])
    model_cfg = config.get("model", None)
    if model_cfg is None:
        return None
    if "use_col_loss" in model_cfg:
        return bool(model_cfg["use_col_loss"])
    pts_head_cfg = model_cfg.get("pts_bbox_head", None)
    if pts_head_cfg is not None and "use_col_loss" in pts_head_cfg:
        return bool(pts_head_cfg["use_col_loss"])
    return None


def _checkpoint_sidecar_config(checkpoint_path: str) -> Optional[Path]:
    checkpoint = Path(checkpoint_path).expanduser()
    if not checkpoint.is_absolute():
        checkpoint = (Path.cwd() / checkpoint).resolve()
    else:
        checkpoint = checkpoint.resolve()
    config_paths = sorted(checkpoint.parent.glob("*.py"))
    if len(config_paths) == 1:
        return config_paths[0]
    return None


def resolve_checkpoint_use_col_loss(checkpoint_path: str) -> Optional[bool]:
    sidecar_config = _checkpoint_sidecar_config(checkpoint_path)
    if sidecar_config is not None:
        use_col_loss = _use_col_loss_from_config(Config.fromfile(str(sidecar_config)))
        if use_col_loss is not None:
            print(f"[collect] checkpoint config {sidecar_config.name}: use_col_loss={use_col_loss}")
            return use_col_loss

    # 只有找不到同目录训练 config 时才读取 checkpoint meta，避免正常路径重复搬 19G checkpoint。
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    meta_config = checkpoint.get("meta", {}).get("config", "")
    match = re.search(r"^\s*use_col_loss\s*=\s*(True|False)\s*$", str(meta_config), flags=re.MULTILINE)
    if match is None:
        return None
    use_col_loss = match.group(1) == "True"
    print(f"[collect] checkpoint meta: use_col_loss={use_col_loss}")
    return use_col_loss


def sync_model_use_col_loss(cfg: Config, use_col_loss: Optional[bool]):
    if use_col_loss is None:
        return
    # 关键调用点：reference 模型结构必须跟 checkpoint 训练信号一致，避免随机 collision 分支进入查表数据。
    cfg.model.use_col_loss = bool(use_col_loss)
    if "pts_bbox_head" in cfg.model:
        cfg.model.pts_bbox_head.use_col_loss = bool(use_col_loss)


def build_sample_records(dataset) -> List[SampleRecord]:
    route_to_pairs: Dict[str, List[tuple]] = defaultdict(list)
    seen_measurements = set()
    for index, info in enumerate(dataset.data_infos):
        frame_id = int(info["frame_idx"])
        measurement_path = os.path.join(
            dataset.data_root,
            info["folder"],
            "anno",
            f"{frame_id:05d}.json.gz",
        )
        route_key = os.path.join(dataset.data_root, info["folder"])
        measurement_path = to_repo_relative(measurement_path)
        route_key = to_repo_relative(route_key)
        if measurement_path in seen_measurements:
            raise RuntimeError(f"duplicated measurement_path in dataset: {measurement_path}")
        seen_measurements.add(measurement_path)
        route_to_pairs[route_key].append((frame_id, index, measurement_path))

    if len(seen_measurements) == 0:
        raise RuntimeError("empty dataset, nothing to collect")

    records: List[SampleRecord] = []
    for route_key in sorted(route_to_pairs.keys()):
        route_pairs = sorted(route_to_pairs[route_key], key=lambda x: x[0])
        for route_frame_index, (frame_id, index, measurement_path) in enumerate(route_pairs):
            records.append(
                SampleRecord(
                    index=index,
                    measurement_path=measurement_path,
                    route_key=route_key,
                    route_id=route_id_from_route_key(route_key),
                    frame_id=frame_id,
                    route_frame_index=route_frame_index,
                    route_frame_count=len(route_pairs),
                )
            )
    return records


def skipped_jsonl_path(output_jsonl: Path) -> Path:
    return output_jsonl.with_name(output_jsonl.name.replace("_all.jsonl", "_skipped.jsonl"))


def load_measurement_paths(jsonl_path: Path, require_loss: bool) -> Set[str]:
    if not jsonl_path.is_file():
        return set()
    measurement_paths: Set[str] = set()
    with jsonl_path.open("r", encoding="utf-8") as fin:
        for line_no, raw_line in enumerate(fin, start=1):
            line = raw_line.strip()
            if line == "":
                continue
            row = json.loads(line)
            measurement_path = row.get("measurement_path", None)
            if measurement_path is None or str(measurement_path).strip() == "":
                raise ValueError(f"{jsonl_path}:{line_no} missing measurement_path")
            if require_loss and "loss_driving_full" not in row:
                raise ValueError(f"{jsonl_path}:{line_no} missing loss_driving_full")
            measurement_path = to_repo_relative(str(measurement_path))
            if measurement_path in measurement_paths:
                raise ValueError(f"{jsonl_path}:{line_no} duplicated measurement_path={measurement_path}")
            measurement_paths.add(measurement_path)
    return measurement_paths


def validate_existing_paths(existing_paths: Set[str], record_by_measurement: Dict[str, SampleRecord], jsonl_path: Path):
    missing_paths = sorted(existing_paths - set(record_by_measurement.keys()))
    if missing_paths:
        raise ValueError(f"{jsonl_path} contains measurement_path not in current dataset: {missing_paths[0]}")


def load_finished_from_done_jsonls(done_jsonls: Sequence[str], record_by_measurement: Dict[str, SampleRecord]):
    done_measurements: Set[str] = set()
    skipped_measurements: Set[str] = set()
    for done_jsonl_arg in done_jsonls:
        done_jsonl = Path(done_jsonl_arg)
        done_paths = load_measurement_paths(done_jsonl, require_loss=True)
        skip_jsonl = skipped_jsonl_path(done_jsonl)
        skip_paths = load_measurement_paths(skip_jsonl, require_loss=False)
        validate_existing_paths(done_paths, record_by_measurement, done_jsonl)
        validate_existing_paths(skip_paths, record_by_measurement, skip_jsonl)
        duplicated = done_measurements & done_paths
        if duplicated:
            raise ValueError(f"duplicated measurement_path across done jsonls: {sorted(duplicated)[0]}")
        done_measurements.update(done_paths)
        skipped_measurements.update(skip_paths)
    return done_measurements, skipped_measurements


def select_route_shard(
    partition_records: Sequence[SampleRecord],
    pending_records: Sequence[SampleRecord],
    num_shards: int,
    shard_id: int,
):
    if num_shards <= 0:
        raise ValueError(f"num_route_shards must be >= 1, got {num_shards}")
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError(f"route_shard_id must be in [0, {num_shards}), got {shard_id}")
    if num_shards == 1:
        return list(pending_records), len(partition_records)

    route_to_count: Dict[str, int] = defaultdict(int)
    for record in partition_records:
        route_to_count[record.route_key] += 1

    # 关键调用点：分片只按完整 route 分配，避免多进程破坏 scene-aware history。
    shard_loads = [0 for _ in range(num_shards)]
    route_to_shard: Dict[str, int] = {}
    for route_key, route_count in sorted(route_to_count.items(), key=lambda item: (-item[1], item[0])):
        target_shard = min(range(num_shards), key=lambda idx: (shard_loads[idx], idx))
        route_to_shard[route_key] = target_shard
        shard_loads[target_shard] += route_count

    return [record for record in pending_records if route_to_shard[record.route_key] == shard_id], shard_loads[shard_id]


def _align_head_temporal_memory(head, previous_route_keys: Optional[List[str]], current_route_keys: List[str]):
    if previous_route_keys is None or previous_route_keys == current_route_keys:
        return
    if getattr(head, "memory_embedding", None) is None:
        return

    previous_pos = {route_key: idx for idx, route_key in enumerate(previous_route_keys)}
    memory_attrs = [
        "memory_embedding",
        "memory_reference_point",
        "memory_timestamp",
        "memory_egopose",
        "memory_velo",
        "sample_time",
        "memory_canbus",
        "his_memory_canbus_len",
        "memory_scene_query",
        "scene_memory_timestamp",
        "memory_mask",
    ]
    for attr in memory_attrs:
        value = getattr(head, attr, None)
        if not torch.is_tensor(value) or value.dim() == 0 or value.size(0) != len(previous_route_keys):
            continue
        rows = []
        for route_key in current_route_keys:
            previous_idx = previous_pos.get(route_key, None)
            if previous_idx is None:
                rows.append(torch.zeros_like(value[:1]))
            else:
                rows.append(value[previous_idx:previous_idx + 1])
        setattr(head, attr, torch.cat(rows, dim=0))

    memory_scene_tokens = getattr(head, "memory_scene_tokens", None)
    if isinstance(memory_scene_tokens, list) and len(memory_scene_tokens) == len(previous_route_keys):
        head.memory_scene_tokens = [
            memory_scene_tokens[previous_pos[route_key]] if route_key in previous_pos else ""
            for route_key in current_route_keys
        ]


def align_model_temporal_memory(model, previous_route_keys: Optional[List[str]], current_route_keys: List[str]):
    # 关键调用点：fixed-slot 调度在 route 结束时会压缩 slot，collection 需要同步压缩模型缓存。
    if previous_route_keys is None or previous_route_keys == current_route_keys:
        return
    if getattr(model, "pts_bbox_head", None) is not None:
        _align_head_temporal_memory(model.pts_bbox_head, previous_route_keys, current_route_keys)
    if getattr(model, "map_head", None) is not None:
        _align_head_temporal_memory(model.map_head, previous_route_keys, current_route_keys)


def build_resume_warmup_records(
    records: Sequence[SampleRecord],
    done_measurements: Set[str],
    pending_measurements: Set[str],
    history_len: int,
) -> List[SampleRecord]:
    if len(done_measurements) == 0 or len(pending_measurements) == 0:
        return []

    route_to_records: Dict[str, List[SampleRecord]] = defaultdict(list)
    for record in records:
        route_to_records[record.route_key].append(record)

    warmup_records: List[SampleRecord] = []
    history_len = max(0, int(history_len))
    for route_key in sorted(route_to_records.keys()):
        route_records = sorted(route_to_records[route_key], key=lambda x: x.frame_id)
        first_pending_pos = None
        for pos, record in enumerate(route_records):
            if record.measurement_path in pending_measurements:
                first_pending_pos = pos
                break
        if first_pending_pos is None or first_pending_pos == 0 or history_len == 0:
            continue

        route_warmup: List[SampleRecord] = []
        expected_frame_id = int(route_records[first_pending_pos].frame_id) - 1
        pos = first_pending_pos - 1
        while pos >= 0 and len(route_warmup) < history_len:
            record = route_records[pos]
            if record.measurement_path not in done_measurements or int(record.frame_id) != expected_frame_id:
                break
            route_warmup.append(record)
            expected_frame_id -= 1
            pos -= 1
        warmup_records.extend(reversed(route_warmup))
    return warmup_records


def build_route_batches(records: Sequence[SampleRecord], route_batch_size: int) -> Iterable[List[SampleRecord]]:
    if route_batch_size <= 0:
        raise ValueError(f"route_batch_size must be >= 1, got {route_batch_size}")

    route_to_records = defaultdict(list)
    for record in records:
        route_to_records[record.route_key].append(record)

    route_order = sorted(route_to_records.keys())
    for route_key in route_order:
        route_to_records[route_key] = deque(sorted(route_to_records[route_key], key=lambda x: x.frame_id))

    pending_routes = deque(route_order)
    active_routes = []
    for _ in range(min(route_batch_size, len(pending_routes))):
        active_routes.append(pending_routes.popleft())

    while active_routes:
        batch_records = []
        next_active_routes = []
        for route_key in active_routes:
            batch_records.append(route_to_records[route_key].popleft())
            if route_to_records[route_key]:
                next_active_routes.append(route_key)
            elif pending_routes:
                next_active_routes.append(pending_routes.popleft())

        route_set = {record.route_key for record in batch_records}
        if len(route_set) != len(batch_records):
            raise RuntimeError("batch contains duplicated route_key")
        yield batch_records
        active_routes = next_active_routes


def move_to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if hasattr(value, "to") and value.__class__.__module__.startswith("mmcv."):
        return value.to(device)
    return value


def collate_collect_samples(samples):
    # 关键调用点：保留 dataset index，后续写 JSONL 时按 route scheduler 的 record 精确对齐。
    skipped = [sample for sample in samples if isinstance(sample, dict) and sample.get("_collect_skip", False)]
    normal_samples = [sample for sample in samples if not (isinstance(sample, dict) and sample.get("_collect_skip", False))]
    if len(normal_samples) == 0:
        return {"_collect_indices": [], "_collect_skips": skipped}
    indices, examples = zip(*normal_samples)
    batch = collate(list(examples), samples_per_gpu=len(examples))
    batch["_collect_indices"] = list(indices)
    batch["_collect_skips"] = skipped
    return batch


def build_collect_dataloader(
    dataset,
    records: Sequence[SampleRecord],
    route_batch_size: int,
    num_workers: int,
    prefetch_factor: int,
):
    dataloader_kwargs = dict(
        dataset=DirectIndexDataset(dataset),
        batch_sampler=RouteBatchSampler(records, route_batch_size),
        collate_fn=collate_collect_samples,
        num_workers=num_workers,
        pin_memory=False,
    )
    if num_workers > 0:
        dataloader_kwargs["persistent_workers"] = True
        dataloader_kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(**dataloader_kwargs)


def resolve_num_workers(cfg: Config, args) -> int:
    num_workers = int(args.num_workers if args.num_workers is not None else cfg.data.get("workers_per_gpu", 0))
    if num_workers < 0:
        raise ValueError(f"num_workers must be >= 0, got {num_workers}")
    if args.prefetch_factor <= 0:
        raise ValueError(f"prefetch_factor must be >= 1, got {args.prefetch_factor}")
    return num_workers


def current_meta(sample_img_metas):
    current_key = max(sample_img_metas.keys())
    return sample_img_metas[current_key]


def assert_batch_matches_records(batch, batch_records: Sequence[SampleRecord]):
    img_metas = batch["img_metas"]
    if len(img_metas) != len(batch_records):
        raise RuntimeError(f"img_metas batch size mismatch: {len(img_metas)} vs {len(batch_records)}")
    for img_meta, record in zip(img_metas, batch_records):
        meta = current_meta(img_meta)
        measurement_path = to_repo_relative(meta["measurement_path"])
        if measurement_path != record.measurement_path:
            raise RuntimeError(
                "route scheduler/sample mismatch: "
                f"meta={measurement_path}, record={record.measurement_path}"
            )


def write_rows(fout, batch_records: Sequence[SampleRecord], loss_dict: Dict[str, torch.Tensor]):
    loss_cpu = {key: value.detach().float().cpu().tolist() for key, value in loss_dict.items()}
    for batch_idx, record in enumerate(batch_records):
        row = {
            "measurement_path": record.measurement_path,
            "route_key": record.route_key,
            "route_id": record.route_id,
            "frame_id": record.frame_id,
            "route_frame_index": record.route_frame_index,
            "route_frame_count": record.route_frame_count,
            "loss_driving_full": float(loss_cpu["loss_driving_full"][batch_idx]),
            "vlm_loss": float(loss_cpu["vlm_loss"][batch_idx]),
            "loss_plan_reg": float(loss_cpu["loss_plan_reg"][batch_idx]),
            "loss_plan_bound": float(loss_cpu["loss_plan_bound"][batch_idx]),
            "loss_plan_col": float(loss_cpu["loss_plan_col"][batch_idx]),
            "loss_vae_gen": float(loss_cpu["loss_vae_gen"][batch_idx]),
        }
        fout.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_skipped_rows(fout, skipped_items: Sequence[dict], record_by_index: Dict[int, SampleRecord]) -> int:
    for item in skipped_items:
        record = record_by_index[int(item["_collect_index"])]
        row = {
            "measurement_path": record.measurement_path,
            "route_key": record.route_key,
            "route_id": record.route_id,
            "frame_id": record.frame_id,
            "route_frame_index": record.route_frame_index,
            "route_frame_count": record.route_frame_count,
            "reason": str(item["_collect_reason"]),
        }
        fout.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(skipped_items)


def main():
    args = parse_args()
    output_jsonl = Path(args.output_jsonl)
    if not output_jsonl.name.endswith("_all.jsonl"):
        raise ValueError(f"output_jsonl must end with _all.jsonl, got {output_jsonl}")
    if not torch.cuda.is_available():
        raise RuntimeError("full-budget collection requires CUDA")

    cfg = Config.fromfile(args.config)
    sync_model_use_col_loss(cfg, resolve_checkpoint_use_col_loss(args.checkpoint))
    set_random_seed(0, deterministic=True)
    torch.backends.cudnn.benchmark = False
    if cfg.get("close_tf32", False):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    train_data_cfg = disable_train_randomness(cfg.data.train)
    dataset = build_dataset(train_data_cfg)
    records = build_sample_records(dataset)
    record_by_index = {record.index: record for record in records}
    record_by_measurement = {record.measurement_path: record for record in records}
    skip_jsonl = skipped_jsonl_path(output_jsonl)
    done_measurements, existing_skipped_measurements = load_finished_from_done_jsonls(
        args.done_jsonl,
        record_by_measurement,
    )
    external_finished_measurements = done_measurements | existing_skipped_measurements
    output_done_measurements = load_measurement_paths(output_jsonl, require_loss=True) if args.resume else set()
    output_skipped_measurements = load_measurement_paths(skip_jsonl, require_loss=False) if args.resume else set()
    validate_existing_paths(output_done_measurements, record_by_measurement, output_jsonl)
    validate_existing_paths(output_skipped_measurements, record_by_measurement, skip_jsonl)
    duplicated_done = done_measurements & output_done_measurements
    if duplicated_done:
        raise ValueError(f"output_jsonl duplicates done-jsonl measurement_path: {sorted(duplicated_done)[0]}")
    done_measurements.update(output_done_measurements)
    existing_skipped_measurements.update(output_skipped_measurements)
    duplicated_finished = done_measurements & existing_skipped_measurements
    if duplicated_finished:
        raise ValueError(f"measurement_path appears in both output and skipped jsonl: {sorted(duplicated_finished)[0]}")

    finished_measurements = done_measurements | existing_skipped_measurements
    shard_partition_records = [
        record for record in records
        if record.measurement_path not in external_finished_measurements
    ]
    all_pending_records = [record for record in records if record.measurement_path not in finished_measurements]
    pending_records, shard_total_records = select_route_shard(
        partition_records=shard_partition_records,
        pending_records=all_pending_records,
        num_shards=args.num_route_shards,
        shard_id=args.route_shard_id,
    )
    warmup_history_len = 2 * int(getattr(dataset, "sample_interval", 1)) + 1
    warmup_records = build_resume_warmup_records(
        records=records,
        done_measurements=done_measurements,
        pending_measurements={record.measurement_path for record in pending_records},
        history_len=warmup_history_len,
    ) if args.resume else []
    print(
        "[collect] resume: "
        f"enabled={args.resume}, "
        f"existing_done={len(done_measurements)}, "
        f"existing_skipped={len(existing_skipped_measurements)}, "
        f"warmup={len(warmup_records)}, "
        f"global_remaining={len(all_pending_records)}, "
        f"shard={args.route_shard_id}/{args.num_route_shards}, "
        f"shard_total={shard_total_records}, "
        f"shard_remaining={len(pending_records)}"
    )
    if len(pending_records) == 0:
        print(f"[collect] nothing to collect, total records already finished: {len(records)}")
        return

    num_workers = resolve_num_workers(cfg, args)
    warmup_dataloader = build_collect_dataloader(
        dataset=dataset,
        records=warmup_records,
        route_batch_size=args.route_batch_size,
        num_workers=0,
        prefetch_factor=args.prefetch_factor,
    ) if warmup_records else None
    dataloader = build_collect_dataloader(
        dataset=dataset,
        records=pending_records,
        route_batch_size=args.route_batch_size,
        num_workers=num_workers,
        prefetch_factor=args.prefetch_factor,
    )
    # 关键调用点：先启动 CPU worker，再初始化 CUDA 模型，避免 worker 继承 CUDA 上下文。
    warmup_iter = iter(warmup_dataloader) if warmup_dataloader is not None else iter(())
    data_iter = iter(dataloader)
    print(
        "[collect] dataloader: "
        f"route_batch_size={args.route_batch_size}, "
        f"num_workers={num_workers}, "
        f"prefetch_factor={args.prefetch_factor if num_workers > 0 else 'disabled'}"
    )

    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu")
    model.CLASSES = checkpoint.get("meta", {}).get("CLASSES", dataset.CLASSES)
    model.cuda()
    model.eval()
    if model.adaption_assigner is not None:
        model.adaption_assigner.reset_sceneaware_history()

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    warmup_skipped = 0
    previous_route_keys = None
    for batch in warmup_iter:
        skipped_items = batch.pop("_collect_skips", [])
        if skipped_items:
            warmup_skipped += len(skipped_items)
        collect_indices = batch.pop("_collect_indices", [])
        if len(collect_indices) == 0:
            continue
        batch_records = [record_by_index[int(index)] for index in collect_indices]
        assert_batch_matches_records(batch, batch_records)
        current_route_keys = [record.route_key for record in batch_records]
        align_model_temporal_memory(model, previous_route_keys, current_route_keys)
        batch = move_to_device(batch, torch.device("cuda"))
        with torch.no_grad():
            model.collect_full_budget_losses(
                forced_budget_value=1.0,
                **batch,
            )
        previous_route_keys = current_route_keys
    if warmup_records:
        print(f"[collect] warmup finished: forwarded={len(warmup_records) - warmup_skipped}, skipped={warmup_skipped}")

    output_mode = "a" if args.resume and output_jsonl.is_file() else "w"
    skipped_mode = "a" if args.resume and skip_jsonl.is_file() else "w"
    written = 0
    skipped = 0
    with output_jsonl.open(output_mode, encoding="utf-8") as fout, skip_jsonl.open(skipped_mode, encoding="utf-8") as fskip:
        with tqdm(
            total=len(finished_measurements) + len(pending_records),
            initial=len(finished_measurements),
            desc="collect full-budget losses",
        ) as progress:
            for batch in data_iter:
                skipped_items = batch.pop("_collect_skips", [])
                if skipped_items:
                    skipped += write_skipped_rows(fskip, skipped_items, record_by_index)
                    progress.update(len(skipped_items))
                    fskip.flush()
                collect_indices = batch.pop("_collect_indices")
                if len(collect_indices) == 0:
                    continue
                batch_records = [record_by_index[int(index)] for index in collect_indices]
                assert_batch_matches_records(batch, batch_records)
                current_route_keys = [record.route_key for record in batch_records]
                align_model_temporal_memory(model, previous_route_keys, current_route_keys)
                batch = move_to_device(batch, torch.device("cuda"))
                with torch.no_grad():
                    loss_dict = model.collect_full_budget_losses(
                        forced_budget_value=1.0,
                        **batch,
                    )
                previous_route_keys = current_route_keys
                write_rows(fout, batch_records, loss_dict)
                written += len(batch_records)
                progress.update(len(batch_records))
                fout.flush()

    finished = len(finished_measurements) + written + skipped
    expected_finished = len(finished_measurements) + len(pending_records)
    if finished != expected_finished:
        raise RuntimeError(f"output row count mismatch: finished={finished}, expected={expected_finished}")
    print(
        "[collect] finished: "
        f"existing={len(done_measurements)}, "
        f"existing_skipped={len(existing_skipped_measurements)}, "
        f"written={written}, "
        f"skipped={skipped}, "
        f"output={output_jsonl}, "
        f"skipped_jsonl={skip_jsonl}"
    )


if __name__ == "__main__":
    main()
