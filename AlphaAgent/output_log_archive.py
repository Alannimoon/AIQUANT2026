import argparse
import importlib.util
import pickle
from pathlib import Path


CATEGORIES = ("momentum", "reversal", "volatility", "volume-price", "cross-sectional")
DEPTH_BINS = (1, 2, 3, 4, 5)
QUALITY_METRIC = "Rank IC"
_FACTOR_AST_MODULE = None
LIGHT_SNAPSHOT_STATE_FILE = ".elite_archive_light_snapshot.pkl"


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretty-print the latest EliteAlpha archive.")
    parser.add_argument("--log-dir", default="log", help="Log directory to scan.")
    parser.add_argument("--history", action="store_true", help="Show latest archive update history instead of archive state.")
    parser.add_argument("--light", action="store_true", help="Show matrix plus only changed cell details.")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    if args.light and not args.history:
        records, source, previous_records, previous_source = load_latest_archive_with_previous(log_dir)
    else:
        records, source = load_latest_history(log_dir) if args.history else load_latest_archive(log_dir)
        previous_records, previous_source = [], None

    if not records:
        print(f"No archive records found under {log_dir}")
        return

    if args.light and not args.history:
        state_records, state_source = load_light_snapshot_state(log_dir)
        if state_records:
            previous_records = state_records
            previous_source = state_source
        changed_keys = find_changed_cells(records, previous_records)
        detail_records = sort_records_by_validation_priority(records)
        detail_numbers = build_detail_numbers(detail_records)
        changed_records = [record for record in detail_records if record_cell_key(record) in changed_keys]
        print(f"source: {source}")
        print(f"previous_source: {previous_source}")
        print(f"quality metric: train {QUALITY_METRIC}")
        print(f"records: {len(records)}")
        print(f"changed cells: {len(changed_records)}")
        print()
        print_matrix(records, changed_keys=changed_keys, detail_numbers=detail_numbers)
        print()
        print_validation_priority(records)
        print()
        print_details(
            changed_records,
            show_history=False,
            title="Changed Details",
            detail_numbers=detail_numbers,
            empty_message="(no archive cell changed this round)",
        )
        save_light_snapshot_state(log_dir, records, source)
        return

    print(f"source: {source}")
    print(f"quality metric: train {QUALITY_METRIC}")
    if args.history:
        matrix_records = reconstruct_archive_from_history(records)
        detail_records = records
        matrix_detail_numbers = build_detail_numbers(matrix_records)
        detail_numbers = None
        print(f"history attempts: {len(records)}")
        print(f"accepted archive records: {len(matrix_records)}")
    else:
        matrix_records = records
        detail_records = sort_records_by_validation_priority(records)
        detail_numbers = build_detail_numbers(detail_records)
        matrix_detail_numbers = detail_numbers
        print(f"records: {len(records)}")
    print()
    print_matrix(matrix_records, detail_numbers=matrix_detail_numbers)
    print()
    print_validation_priority(matrix_records)
    print()
    print_details(detail_records, show_history=args.history, detail_numbers=detail_numbers)


def list_archive_files(log_dir: Path) -> list[Path]:
    return sorted(
        log_dir.glob("*/elite archive/*/*.pkl"),
        key=lambda path: (path.stat().st_mtime, str(path)),
    )


def list_history_files(log_dir: Path) -> list[Path]:
    return sorted(
        log_dir.glob("*/elite archive history/*/*.pkl"),
        key=lambda path: (path.stat().st_mtime, str(path)),
    )


def load_latest_archive(log_dir: Path) -> tuple[list[dict], Path | None]:
    files = list_archive_files(log_dir)
    if files:
        source = files[-1]
        return load_pickle_records(source), source

    history_records, source = load_latest_history(log_dir)
    if not history_records:
        return [], source
    return reconstruct_archive_from_history(history_records), source


def load_latest_archive_with_previous(log_dir: Path) -> tuple[list[dict], Path | None, list[dict], Path | None]:
    files = list_archive_files(log_dir)
    if files:
        source = files[-1]
        previous_source = files[-2] if len(files) >= 2 else None
        previous_records = load_pickle_records(previous_source) if previous_source else []
        return load_pickle_records(source), source, previous_records, previous_source

    history_files = list_history_files(log_dir)
    if not history_files:
        return [], None, [], None

    source = history_files[-1]
    previous_source = history_files[-2] if len(history_files) >= 2 else None
    records = reconstruct_archive_from_history(load_pickle_records(source))
    previous_records = reconstruct_archive_from_history(load_pickle_records(previous_source)) if previous_source else []
    return records, source, previous_records, previous_source


def load_light_snapshot_state(log_dir: Path) -> tuple[list[dict], str | None]:
    state_path = log_dir / LIGHT_SNAPSHOT_STATE_FILE
    if not state_path.exists():
        return [], None
    try:
        with state_path.open("rb") as f:
            state = pickle.load(f)
    except Exception:
        return [], None

    records = state.get("records", []) if isinstance(state, dict) else []
    source = state.get("source") if isinstance(state, dict) else None
    if not isinstance(records, list):
        return [], source
    return records, source


def save_light_snapshot_state(log_dir: Path, records: list[dict], source: Path | None) -> None:
    state_path = log_dir / LIGHT_SNAPSHOT_STATE_FILE
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with state_path.open("wb") as f:
        pickle.dump(
            {
                "source": None if source is None else str(source),
                "records": records,
            },
            f,
        )


def load_latest_history(log_dir: Path) -> tuple[list[dict], Path | None]:
    files = list_history_files(log_dir)
    if not files:
        return [], None
    source = files[-1]
    return load_pickle_records(source), source


def load_pickle_records(path: Path) -> list[dict]:
    with path.open("rb") as f:
        records = pickle.load(f)
    return [record_to_dict(record) for record in records]


def record_to_dict(record) -> dict:
    if isinstance(record, dict):
        data = dict(record)
    elif hasattr(record, "to_dict"):
        data = record.to_dict()
    else:
        raise TypeError(f"Unsupported archive record type: {type(record)!r}")

    metric = data.get("factor_complexity_metric") or "depth"
    data["factor_complexity_metric"] = metric
    data["quality_metric"] = data.get("quality_metric") or QUALITY_METRIC
    if data.get("factor_complexity_value") is None:
        data["factor_complexity_value"] = calculate_ast_metric(data.get("factor_expression"), metric)

    if metric == "depth" and data.get("factor_ast_depth") is None:
        data["factor_ast_depth"] = data.get("factor_complexity_value")
    elif metric == "vertex" and data.get("factor_ast_node_count") is None:
        data["factor_ast_node_count"] = data.get("factor_complexity_value")
    return data


def reconstruct_archive_from_history(history_records: list[dict]) -> list[dict]:
    cells = {}
    for record in history_records:
        if not record.get("accepted"):
            continue
        key = record_cell_key(record)
        cells[key] = record
    return list(cells.values())


def find_changed_cells(records: list[dict], previous_records: list[dict]) -> set[tuple[str, int]]:
    if not previous_records:
        return {record_cell_key(record) for record in records}

    previous_by_cell = {record_cell_key(record): record_signature(record) for record in previous_records}
    changed = set()
    for record in records:
        key = record_cell_key(record)
        if previous_by_cell.get(key) != record_signature(record):
            changed.add(key)
    return changed


def record_cell_key(record: dict) -> tuple[str, int]:
    return record.get("category"), int(record.get("depth_bin"))


def record_signature(record: dict) -> tuple:
    return (
        record.get("factor_name"),
        record.get("factor_expression"),
        record.get("category"),
        int(record.get("depth_bin")),
        normalize_signature_float(record.get("quality")),
        normalize_signature_float(record.get("validation_Rank IC")),
    )


def normalize_signature_float(value):
    try:
        return round(float(value), 12)
    except (TypeError, ValueError):
        return value


def build_detail_numbers(records: list[dict]) -> dict[tuple[str, int], int]:
    return {record_cell_key(record): idx for idx, record in enumerate(records, start=1)}


def print_matrix(
    records: list[dict],
    changed_keys: set[tuple[str, int]] | None = None,
    detail_numbers: dict[tuple[str, int], int] | None = None,
) -> None:
    changed_keys = changed_keys or set()
    cells = {}
    detail_numbers = detail_numbers or build_detail_numbers(records)
    for idx, record in enumerate(records, start=1):
        key = record_cell_key(record)
        cells[key] = record

    label_width = max(len(category) for category in CATEGORIES)
    cell_width = 18
    header = " " * (label_width + 2) + "".join(f"bin={d}".center(cell_width) for d in DEPTH_BINS)
    print("Archive Matrix")
    print(header)
    print("-" * len(header))
    for category in CATEGORIES:
        top_row = [f"{category:<{label_width}}  "]
        bottom_row = [" " * (label_width + 2)]
        has_record = False
        for depth in DEPTH_BINS:
            record = cells.get((category, depth))
            if record is None:
                top_cell = "."
                bottom_cell = ""
            else:
                has_record = True
                number = detail_numbers[(category, depth)]
                top_cell = f"[[{number}]]" if (category, depth) in changed_keys else f"[{number}]"
                bottom_cell = f"quality={format_quality(record.get('quality'))}"
            top_row.append(top_cell.center(cell_width))
            bottom_row.append(bottom_cell.center(cell_width))
        print("".join(top_row))
        if has_record:
            print("".join(bottom_row))


def print_details(
    records: list[dict],
    *,
    show_history: bool,
    title: str = "Details",
    detail_numbers: dict[tuple[str, int], int] | None = None,
    empty_message: str = "(empty)",
) -> None:
    print(title)
    if not records:
        print(empty_message)
        return

    for idx, record in enumerate(records, start=1):
        number = detail_numbers.get(record_cell_key(record), idx) if detail_numbers else idx
        accepted = record.get("accepted")
        accepted_text = "" if accepted is None else f" | accepted={accepted}"
        incumbent = record.get("incumbent_factor_name")
        incumbent_text = "" if not incumbent else f" | incumbent={incumbent} ({record.get('incumbent_quality')})"
        optional_stats = []
        if record.get("factor_ast_depth") is not None:
            optional_stats.append(f"ast_depth={record.get('factor_ast_depth')}")
        if record.get("factor_ast_node_count") is not None:
            optional_stats.append(f"ast_nodes={record.get('factor_ast_node_count')}")
        optional_text = "" if not optional_stats else " | " + " | ".join(optional_stats)
        quality_metric = record.get("quality_metric") or QUALITY_METRIC
        validation_text = ""
        if record.get("validation_Rank IC") is not None:
            validation_text = f" | validation_Rank IC={format_quality(record.get('validation_Rank IC'))}"
        print(
            f"[{number}] {record.get('factor_name')} "
            f"| cell=({record.get('category')}, {record.get('depth_bin')}) "
            f"| metric={record.get('factor_complexity_metric')} "
            f"| metric_value={record.get('factor_complexity_value')}"
            f"{optional_text} "
            f"| quality_metric=train {quality_metric} "
            f"| quality={record.get('quality')}{accepted_text}{incumbent_text}"
            f"{validation_text}"
        )
        expression = record.get("factor_expression")
        if expression:
            print(f"    expr: {expression}")
        description = record.get("factor_description")
        if description and not show_history:
            print(f"    desc: {shorten(description, 180)}")
        print()


def print_validation_priority(records: list[dict]) -> None:
    print("Validation Priority (no corr/depth regularization)")
    if not records:
        print("(empty)")
        return

    ranked = []
    for record in records:
        validation_rank_ic = record.get("validation_Rank IC")
        try:
            validation_rank_ic = float(validation_rank_ic)
        except (TypeError, ValueError):
            validation_rank_ic = None
        ranked.append((validation_rank_ic, record))

    ranked.sort(
        key=lambda item: (
            item[0] is not None,
            item[0] if item[0] is not None else float("-inf"),
        ),
        reverse=True,
    )
    for idx, (validation_rank_ic, record) in enumerate(ranked, start=1):
        print(
            f"[{idx}] {record.get('factor_name')} "
            f"| validation_Rank IC={format_quality(validation_rank_ic)} "
            f"| train_Rank IC={format_quality(record.get('quality'))} "
            f"| cell=({record.get('category')}, {record.get('depth_bin')})"
        )


def sort_records_by_validation_priority(records: list[dict]) -> list[dict]:
    return sorted(
        records,
        key=lambda record: (
            parse_optional_float(record.get("validation_Rank IC")) is not None,
            parse_optional_float(record.get("validation_Rank IC"))
            if parse_optional_float(record.get("validation_Rank IC")) is not None
            else float("-inf"),
            parse_optional_float(record.get("quality"))
            if parse_optional_float(record.get("quality")) is not None
            else float("-inf"),
        ),
        reverse=True,
    )


def parse_optional_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def shorten(text: str, max_len: int) -> str:
    text = str(text).replace("\n", " ").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def format_quality(value, digits: int = 5) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def calculate_ast_metric(expression: str | None, metric: str):
    if not expression:
        return None
    try:
        mod = load_factor_ast_module()
        node = mod.parse_expression(expression)
        if metric == "vertex":
            return node_count(node)
        return node_depth(node)
    except Exception:
        return None


def load_factor_ast_module():
    global _FACTOR_AST_MODULE
    if _FACTOR_AST_MODULE is not None:
        return _FACTOR_AST_MODULE

    path = Path(__file__).parent / "alphaagent" / "components" / "coder" / "factor_coder" / "factor_ast.py"
    spec = importlib.util.spec_from_file_location("factor_ast_for_archive_view", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _FACTOR_AST_MODULE = module
    return module


def node_depth(node) -> int:
    node_type = node.__class__.__name__
    if node_type == "FunctionNode":
        if not node.args:
            return 1
        return 1 + max(node_depth(arg) for arg in node.args)
    if node_type == "BinaryOpNode":
        return 1 + max(node_depth(node.left), node_depth(node.right))
    if node_type == "ConditionalNode":
        return 1 + max(
            node_depth(node.condition),
            node_depth(node.true_expr),
            node_depth(node.false_expr),
        )
    return 1


def node_count(node) -> int:
    node_type = node.__class__.__name__
    if node_type == "FunctionNode":
        return 1 + sum(node_count(arg) for arg in node.args)
    if node_type == "BinaryOpNode":
        return 1 + node_count(node.left) + node_count(node.right)
    if node_type == "ConditionalNode":
        return 1 + node_count(node.condition) + node_count(node.true_expr) + node_count(node.false_expr)
    return 1


if __name__ == "__main__":
    main()
