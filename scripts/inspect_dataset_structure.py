from pathlib import Path
import argparse
import json
import pandas as pd


def build_tree(root: Path) -> list[str]:
    """
    建立資料夾與檔案的樹狀結構。
    只列出資料夾與常見資料檔，不輸出資料內容。
    """
    lines = []

    allowed_suffixes = {
        ".csv",
        ".txt",
        ".tsv",
        ".json",
        ".md",
    }

    def should_show(path: Path) -> bool:
        if path.is_dir():
            return True
        return path.suffix.lower() in allowed_suffixes

    def walk(path: Path, prefix: str = ""):
        entries = sorted(
            [p for p in path.iterdir() if should_show(p)],
            key=lambda p: (p.is_file(), p.name.lower())
        )

        for i, entry in enumerate(entries):
            is_last = i == len(entries) - 1
            connector = "└── " if is_last else "├── "
            lines.append(f"{prefix}{connector}{entry.name}")

            if entry.is_dir():
                extension = "    " if is_last else "│   "
                walk(entry, prefix + extension)

    lines.append(root.name)
    walk(root)
    return lines


def read_csv_header(csv_path: Path, sample_rows: int = 100) -> dict:
    """
    讀取 CSV 欄位名稱與推測 dtype。
    注意：報告不會輸出任何資料列內容。
    """
    info = {
        "path": csv_path.as_posix(),
        "relative_path": None,
        "filename": csv_path.name,
        "columns": [],
        "column_count": 0,
        "dtypes": {},
        "error": None,
    }

    encodings = ["utf-8", "utf-8-sig", "latin1", "cp1252"]

    last_error = None

    for encoding in encodings:
        try:
            header_df = pd.read_csv(
                csv_path,
                nrows=0,
                encoding=encoding,
                low_memory=False,
            )

            info["columns"] = list(header_df.columns)
            info["column_count"] = len(header_df.columns)

            if sample_rows > 0:
                sample_df = pd.read_csv(
                    csv_path,
                    nrows=sample_rows,
                    encoding=encoding,
                    low_memory=False,
                )
                info["dtypes"] = {
                    col: str(dtype)
                    for col, dtype in sample_df.dtypes.items()
                }
            else:
                info["dtypes"] = {
                    col: "unknown"
                    for col in info["columns"]
                }

            info["encoding_used"] = encoding
            return info

        except Exception as e:
            last_error = e

    info["error"] = repr(last_error)
    return info


def collect_csv_schemas(root: Path, sample_rows: int = 100) -> list[dict]:
    csv_files = sorted(root.rglob("*.csv"))
    schemas = []

    for csv_path in csv_files:
        schema = read_csv_header(csv_path, sample_rows=sample_rows)

        try:
            schema["relative_path"] = csv_path.relative_to(root).as_posix()
        except ValueError:
            schema["relative_path"] = csv_path.as_posix()

        schemas.append(schema)

    return schemas


def write_markdown_report(
    root: Path,
    tree_lines: list[str],
    csv_schemas: list[dict],
    output_path: Path,
):
    with output_path.open("w", encoding="utf-8") as f:
        f.write("# Dataset Structure Report\n\n")

        f.write("## Root Directory\n\n")
        f.write("```text\n")
        f.write(root.as_posix())
        f.write("\n```\n\n")

        f.write("## Folder Tree\n\n")
        f.write("```text\n")
        f.write("\n".join(tree_lines))
        f.write("\n```\n\n")

        f.write("## CSV Files Summary\n\n")
        f.write(f"- CSV file count: `{len(csv_schemas)}`\n\n")

        f.write("| # | Relative path | Column count | Encoding | Status |\n")
        f.write("|---:|---|---:|---|---|\n")

        for i, schema in enumerate(csv_schemas, start=1):
            status = "OK" if schema["error"] is None else "ERROR"
            encoding = schema.get("encoding_used", "unknown")
            f.write(
                f"| {i} | `{schema['relative_path']}` "
                f"| {schema['column_count']} "
                f"| `{encoding}` "
                f"| `{status}` |\n"
            )

        f.write("\n")

        f.write("## CSV Schemas\n\n")

        for schema in csv_schemas:
            f.write(f"### `{schema['relative_path']}`\n\n")

            if schema["error"] is not None:
                f.write(f"Error: `{schema['error']}`\n\n")
                continue

            f.write(f"- File name: `{schema['filename']}`\n")
            f.write(f"- Column count: `{schema['column_count']}`\n")
            f.write(f"- Encoding used: `{schema.get('encoding_used', 'unknown')}`\n\n")

            f.write("| # | Column | Inferred dtype |\n")
            f.write("|---:|---|---|\n")

            for i, col in enumerate(schema["columns"], start=1):
                dtype = schema["dtypes"].get(col, "unknown")
                f.write(f"| {i} | `{col}` | `{dtype}` |\n")

            f.write("\n")


def main():
    parser = argparse.ArgumentParser(
        description="Inspect dataset folder tree and CSV schemas without exporting data rows."
    )
    parser.add_argument(
        "--root",
        type=str,
        default="data",
        help="Dataset root directory. Default: data",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="dataset_reports",
        help="Output report directory. Default: dataset_reports",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=100,
        help="Rows used only for dtype inference. Set 0 to read headers only.",
    )

    args = parser.parse_args()

    root = Path(args.root)
    output_dir = Path(args.output_dir)

    if not root.exists():
        raise FileNotFoundError(f"Root directory not found: {root}")

    output_dir.mkdir(parents=True, exist_ok=True)

    tree_lines = build_tree(root)
    csv_schemas = collect_csv_schemas(root, sample_rows=args.sample_rows)

    json_report = {
        "root": root.resolve().as_posix(),
        "csv_file_count": len(csv_schemas),
        "csv_schemas": csv_schemas,
    }

    json_output_path = output_dir / "dataset_schema.json"
    md_output_path = output_dir / "dataset_structure.md"

    with json_output_path.open("w", encoding="utf-8") as f:
        json.dump(json_report, f, indent=2, ensure_ascii=False)

    write_markdown_report(
        root=root,
        tree_lines=tree_lines,
        csv_schemas=csv_schemas,
        output_path=md_output_path,
    )

    print("Dataset inspection completed.")
    print(f"Root: {root}")
    print(f"CSV files found: {len(csv_schemas)}")
    print(f"Markdown report: {md_output_path}")
    print(f"JSON schema: {json_output_path}")


if __name__ == "__main__":
    main()