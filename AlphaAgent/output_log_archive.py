import argparse
import importlib.util
import pickle
from pathlib import Path


CATEGORIES = ("momentum", "reversal", "volatility", "volume-price", "cross-sectional")
DEPTH_BINS = (1, 2, 3, 4, 5)
_FACTOR_AST_MODULE = None


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretty-print the latest EliteAlpha archive.")
    parser.add_argument("--log-dir", default="log", help="Log directory to scan.")
    parser.add_argument("--history", action="store_true", help="Show latest archive update history instead of archive state.")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    records, source = load_latest_history(log_dir) if args.history else load_latest_archive(log_dir)
    if not records:
        print(f"No archive records found under {log_dir}")
        return

    print(f"source: {source}")
    if args.history:
        matrix_records = reconstruct_archive_from_history(records)
        print(f"history attempts: {len(records)}")
        print(f"accepted archive records: {len(matrix_records)}")
    else:
        matrix_records = records
        print(f"records: {len(records)}")
    print()
    print_matrix(matrix_records)
    print()
    print_details(records, show_history=args.history)


def load_latest_archive(log_dir: Path) -> tuple[list[dict], Path | None]:
    files = sorted(log_dir.glob("*/elite archive/*/*.pkl"))
    if files:
        source = files[-1]
        return load_pickle_records(source), source

    history_records, source = load_latest_history(log_dir)
    if not history_records:
        return [], source
    return reconstruct_archive_from_history(history_records), source


def load_latest_history(log_dir: Path) -> tuple[list[dict], Path | None]:
    files = sorted(log_dir.glob("*/elite archive history/*/*.pkl"))
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
        key = (record.get("category"), int(record.get("depth_bin")))
        cells[key] = record
    return list(cells.values())


def print_matrix(records: list[dict]) -> None:
    cells = {}
    detail_numbers = {}
    for idx, record in enumerate(records, start=1):
        key = (record.get("category"), int(record.get("depth_bin")))
        cells[key] = record
        detail_numbers[key] = idx

    label_width = max(len(category) for category in CATEGORIES)
    cell_width = 12
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
                top_cell = f"[{number}]"
                bottom_cell = f"q={format_quality(record.get('quality'))}"
            top_row.append(top_cell.center(cell_width))
            bottom_row.append(bottom_cell.center(cell_width))
        print("".join(top_row))
        if has_record:
            print("".join(bottom_row))


def print_details(records: list[dict], *, show_history: bool) -> None:
    print("Details")
    for idx, record in enumerate(records, start=1):
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
        print(
            f"[{idx}] {record.get('factor_name')} "
            f"| cell=({record.get('category')}, {record.get('depth_bin')}) "
            f"| metric={record.get('factor_complexity_metric')} "
            f"| metric_value={record.get('factor_complexity_value')}"
            f"{optional_text} "
            f"| quality={record.get('quality')}{accepted_text}{incumbent_text}"
        )
        expression = record.get("factor_expression")
        if expression:
            print(f"    expr: {expression}")
        description = record.get("factor_description")
        if description and not show_history:
            print(f"    desc: {shorten(description, 180)}")
        print()


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
