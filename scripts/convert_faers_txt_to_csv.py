from pathlib import Path
import argparse
import pandas as pd


def try_read_txt(path: Path) -> pd.DataFrame:
    """
    FAERS ASCII TXT 通常使用 $ 作為分隔符。
    這裡使用 dtype=str，避免 ID、日期、代碼被 pandas 自動轉型。
    """
    encodings = ["latin1", "utf-8", "utf-8-sig", "cp1252"]
    separators = ["$", "\t", "|", ","]

    last_error = None

    for encoding in encodings:
        for sep in separators:
            try:
                df = pd.read_csv(
                    path,
                    sep=sep,
                    dtype=str,
                    encoding=encoding,
                    low_memory=False,
                )

                # 避免錯誤分隔符造成整行只有一欄
                if len(df.columns) > 1:
                    return df

            except Exception as e:
                last_error = e

    raise RuntimeError(f"Failed to read {path}: {repr(last_error)}")


def convert_one_file(txt_path: Path, overwrite: bool = False) -> dict:
    csv_path = txt_path.with_suffix(".csv")

    result = {
        "txt_path": txt_path.as_posix(),
        "csv_path": csv_path.as_posix(),
        "status": "pending",
        "rows": 0,
        "columns": 0,
        "error": None,
    }

    if csv_path.exists() and not overwrite:
        result["status"] = "skipped_exists"
        return result

    try:
        df = try_read_txt(txt_path)

        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False, encoding="utf-8")

        result["status"] = "converted"
        result["rows"] = len(df)
        result["columns"] = len(df.columns)

    except Exception as e:
        result["status"] = "error"
        result["error"] = repr(e)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Convert FAERS TXT files to CSV files."
    )
    parser.add_argument(
        "--root",
        type=str,
        default="data/2004Q1_2012Q4",
        help="Root folder containing FAERS TXT files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing CSV files.",
    )

    args = parser.parse_args()

    root = Path(args.root)

    if not root.exists():
        raise FileNotFoundError(f"Root folder not found: {root}")

    txt_files = sorted(
        list(root.rglob("*.TXT")) +
        list(root.rglob("*.txt"))
    )

    print(f"TXT files found: {len(txt_files)}")

    converted = 0
    skipped = 0
    errors = 0

    for txt_path in txt_files:
        result = convert_one_file(txt_path, overwrite=args.overwrite)

        if result["status"] == "converted":
            converted += 1
            print(
                f"[OK] {result['txt_path']} -> {result['csv_path']} "
                f"({result['rows']} rows, {result['columns']} columns)"
            )

        elif result["status"] == "skipped_exists":
            skipped += 1
            print(f"[SKIP] {result['csv_path']} already exists")

        else:
            errors += 1
            print(f"[ERROR] {result['txt_path']}: {result['error']}")

    print()
    print("Conversion completed.")
    print(f"Converted: {converted}")
    print(f"Skipped: {skipped}")
    print(f"Errors: {errors}")


if __name__ == "__main__":
    main()