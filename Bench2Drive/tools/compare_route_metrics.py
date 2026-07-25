import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RouteResult:
    score_composed: float
    success: bool


@dataclass(frozen=True)
class LoadedRoutes:
    records: dict
    empty_record_files: list


def parse_route_key(route_id):
    parts = route_id.split("_")
    if len(parts) < 3 or parts[0] != "RouteScenario" or parts[2] != "rep0":
        raise ValueError(f"Unexpected route_id format: {route_id}")
    route_key = parts[1]
    if not route_key.isdigit():
        raise ValueError(f"Unexpected numeric route id in route_id: {route_id}")
    return route_key


def is_success(record):
    if record["status"] not in {"Completed", "Perfect"}:
        return False

    # min_speed_infractions 不影响 Bench2Drive 的 SR 判定。
    for infraction_name, infractions in record["infractions"].items():
        if infraction_name == "min_speed_infractions":
            continue
        if infractions:
            return False
    return True


def load_routes(folder):
    folder = Path(folder)
    if not folder.is_dir():
        raise ValueError(f"Route result folder does not exist: {folder}")

    records = {}
    empty_record_files = []
    for file_path in sorted(folder.glob("*.json")):
        if file_path.name == "merged.json":
            continue

        with file_path.open("r") as f:
            data = json.load(f)

        route_records = data["_checkpoint"]["records"]
        if len(route_records) == 0:
            empty_record_files.append(file_path.name)
            continue
        if len(route_records) != 1:
            raise ValueError(f"{file_path} contains {len(route_records)} records, expected 1")

        record = route_records[0]
        route_key = parse_route_key(record["route_id"])
        if route_key in records:
            raise ValueError(f"Duplicate route id {route_key} in folder {folder}")

        records[route_key] = RouteResult(
            score_composed=float(record["scores"]["score_composed"]),
            success=is_success(record),
        )

    return LoadedRoutes(records=records, empty_record_files=empty_record_files)


def compute_metrics(records, common_route_keys):
    route_count = len(common_route_keys)
    if route_count == 0:
        raise ValueError("No common valid route records between folder A and folder B")

    ds = sum(records[route_key].score_composed for route_key in common_route_keys) / route_count
    success_count = sum(records[route_key].success for route_key in common_route_keys)
    sr = success_count / route_count
    return ds, sr, success_count


def format_list(values):
    if not values:
        return "None"
    return ", ".join(values)


def print_metrics_table(name_a, name_b, routes_a, routes_b, common_route_keys):
    ds_a, sr_a, success_a = compute_metrics(routes_a.records, common_route_keys)
    ds_b, sr_b, success_b = compute_metrics(routes_b.records, common_route_keys)

    print("Metrics on common valid routes")
    print(f"{'name':<20} {'valid_routes':>12} {'common_routes':>13} {'DS':>12} {'SR':>12} {'success_count':>14}")
    print("-" * 87)
    print(f"{name_a:<20} {len(routes_a.records):>12} {len(common_route_keys):>13} {ds_a:>12.6f} {sr_a:>12.6f} {success_a:>14}")
    print(f"{name_b:<20} {len(routes_b.records):>12} {len(common_route_keys):>13} {ds_b:>12.6f} {sr_b:>12.6f} {success_b:>14}")
    print()
    print("Delta (B - A)")
    print(f"{'delta_DS':>12} {'delta_SR':>12}")
    print("-" * 25)
    print(f"{ds_b - ds_a:>12.6f} {sr_b - sr_a:>12.6f}")


def print_coverage_report(name_a, name_b, routes_a, routes_b, common_route_keys):
    only_a = sorted(set(routes_a.records) - set(routes_b.records), key=int)
    only_b = sorted(set(routes_b.records) - set(routes_a.records), key=int)

    print()
    print("Coverage report")
    print(f"{name_a} empty record files ({len(routes_a.empty_record_files)}): {format_list(routes_a.empty_record_files)}")
    print(f"{name_b} empty record files ({len(routes_b.empty_record_files)}): {format_list(routes_b.empty_record_files)}")
    print(f"common valid route ids ({len(common_route_keys)}): {format_list(common_route_keys)}")
    print(f"only {name_a} valid route ids ({len(only_a)}): {format_list(only_a)}")
    print(f"only {name_b} valid route ids ({len(only_b)}): {format_list(only_b)}")


def print_top_a_better_routes(name_a, name_b, routes_a, routes_b, common_route_keys, top_k=5):
    # 这里专门看 A 相对 B 的 DS 优势，用于快速定位最值得回看日志和视频的 route。
    route_deltas = []
    for route_key in common_route_keys:
        score_a = routes_a.records[route_key].score_composed
        score_b = routes_b.records[route_key].score_composed
        route_deltas.append((score_a - score_b, route_key, score_a, score_b))

    route_deltas = sorted(route_deltas, key=lambda item: (-item[0], int(item[1])))
    print()
    print(f"Top {top_k} routes where {name_a} is better than {name_b}")
    print(f"{'route_id':>10} {'A-B_DS':>12} {name_a + '_DS':>12} {name_b + '_DS':>12}")
    print("-" * 52)
    for delta, route_key, score_a, score_b in route_deltas[:top_k]:
        print(f"{route_key:>10} {delta:>12.6f} {score_a:>12.6f} {score_b:>12.6f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Compare local Bench2Drive DS/SR metrics between two routes/res folders.")
    parser.add_argument("--folder-a", required=True, help="Folder A containing Bench2Drive route result json files.")
    parser.add_argument("--folder-b", required=True, help="Folder B containing Bench2Drive route result json files.")
    parser.add_argument("--name-a", default="A", help="Display name for folder A.")
    parser.add_argument("--name-b", default="B", help="Display name for folder B.")
    return parser.parse_args()


def main():
    args = parse_args()
    routes_a = load_routes(args.folder_a)
    routes_b = load_routes(args.folder_b)
    common_route_keys = sorted(set(routes_a.records) & set(routes_b.records), key=int)

    print_metrics_table(args.name_a, args.name_b, routes_a, routes_b, common_route_keys)
    print_top_a_better_routes(args.name_a, args.name_b, routes_a, routes_b, common_route_keys)
    print_coverage_report(args.name_a, args.name_b, routes_a, routes_b, common_route_keys)


if __name__ == "__main__":
    try:
        main()
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
