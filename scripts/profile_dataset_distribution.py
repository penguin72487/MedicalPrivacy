#!/usr/bin/env python3
"""
Build privacy-aware distribution profiles from real CSV datasets.

Reads real CSV locally and exports only aggregate statistics:
- row_count, missing_rate
- numeric quantiles / mean / std / min / max
- categorical top-k value frequencies
- string length summaries
- date-like min/max and format hints

It does NOT export raw rows.
"""
from __future__ import annotations
import argparse, hashlib, json, re
from pathlib import Path
from typing import Any
import pandas as pd

DATE_PATTERNS = ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%m/%d/%Y", "%Y%m%d"]


def parse_args():
    p = argparse.ArgumentParser(description="Profile real CSV distributions without exporting raw rows.")
    p.add_argument("--data-root", default="data")
    p.add_argument("--schema", default="dataset_reports/dataset_schema.json")
    p.add_argument("--output", default="dataset_reports/dataset_distribution_profile.json")
    p.add_argument("--top-k", type=int, default=30)
    p.add_argument("--min-count", type=int, default=5, help="Suppress categories appearing fewer than this count.")
    p.add_argument("--sample-rows", type=int, default=50000, help="0 means read all rows.")
    p.add_argument("--hash-categories", action="store_true", help="Hash category labels in the profile.")
    p.add_argument("--encoding-fallback", default="utf-8,utf-8-sig,latin1,cp1252")
    return p.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_hash(value: str) -> str:
    return "HASH_" + hashlib.sha256(value.encode("utf-8", errors="ignore")).hexdigest()[:16]


def read_csv_safe(path: Path, encoding_hint: str | None, nrows: int, encodings: list[str]):
    candidates = []
    if encoding_hint:
        candidates.append(encoding_hint)
    candidates.extend([e for e in encodings if e not in candidates])
    last_error = None
    for enc in candidates:
        try:
            df = pd.read_csv(path, dtype=str, encoding=enc, low_memory=False,
                             nrows=None if nrows == 0 else nrows)
            return df, enc, None
        except Exception as e:
            last_error = repr(e)
    return None, None, last_error


def infer_date_profile(series: pd.Series):
    s = series.dropna().astype(str)
    s = s[s.str.len() > 0]
    if len(s) == 0:
        return None
    sample = s.head(min(1000, len(s)))
    best_fmt, best_success = None, 0.0
    for fmt in DATE_PATTERNS:
        parsed = pd.to_datetime(sample, format=fmt, errors="coerce")
        rate = float(parsed.notna().mean())
        if rate > best_success:
            best_fmt, best_success = fmt, rate
    parsed = pd.to_datetime(sample, errors="coerce")
    rate = float(parsed.notna().mean())
    if rate > best_success:
        best_fmt, best_success = "infer", rate
    if best_fmt is None or best_success < 0.80:
        return None
    full = pd.to_datetime(s, format=None if best_fmt == "infer" else best_fmt, errors="coerce").dropna()
    if len(full) == 0:
        return None
    return {"detected": True, "format": best_fmt, "parse_success_rate": best_success,
            "min": full.min().strftime("%Y-%m-%d %H:%M:%S"),
            "max": full.max().strftime("%Y-%m-%d %H:%M:%S")}


def numeric_profile(series: pd.Series):
    x = pd.to_numeric(series, errors="coerce").dropna()
    if len(x) == 0:
        return None
    q = x.quantile([0, .01, .05, .25, .50, .75, .95, .99, 1.0])
    return {
        "count": int(len(x)), "min": float(q.loc[0]), "q01": float(q.loc[.01]),
        "q05": float(q.loc[.05]), "q25": float(q.loc[.25]), "median": float(q.loc[.50]),
        "q75": float(q.loc[.75]), "q95": float(q.loc[.95]), "q99": float(q.loc[.99]),
        "max": float(q.loc[1.0]), "mean": float(x.mean()),
        "std": float(x.std(ddof=0)) if len(x) > 1 else 0.0,
        "integer_like": bool((x % 1 == 0).mean() > .98),
    }


def categorical_profile(series: pd.Series, top_k: int, min_count: int, hash_categories: bool):
    s = series.dropna().astype(str)
    s = s[s.str.len() > 0]
    total = int(len(s))
    vc = s.value_counts(dropna=True)
    top_values, kept = [], 0
    for value, count in vc.head(top_k).items():
        count = int(count)
        if count < min_count:
            continue
        top_values.append({
            "value": safe_hash(value) if hash_categories else value,
            "count": count,
            "probability": float(count / total) if total else 0.0,
        })
        kept += count
    lengths = s.str.len() if total else pd.Series(dtype=float)
    return {
        "non_missing_count": total,
        "unique_count": int(vc.shape[0]),
        "top_values": top_values,
        "other_probability": float(max(total - kept, 0) / total) if total else 0.0,
        "string_length": {
            "min": int(lengths.min()) if total else 0,
            "median": float(lengths.median()) if total else 0,
            "max": int(lengths.max()) if total else 0,
        },
        "hashed": bool(hash_categories),
    }


def profile_column(df: pd.DataFrame, column: str, dtype_hint: str, top_k: int, min_count: int, hash_categories: bool):
    series = df[column] if column in df.columns else pd.Series(dtype=str)
    total_rows = len(df)
    missing = series.isna() | (series.astype(str).str.len() == 0)
    missing_rate = float(missing.mean()) if total_rows else 1.0
    prof = {"dtype_hint": dtype_hint, "missing_rate": missing_rate}
    if re.search(r"(date|time|_dt$|dt$|dob|dod)", column, flags=re.I):
        dp = infer_date_profile(series)
        if dp is not None:
            prof.update({"kind": "date", "date": dp})
            return prof
    npf = numeric_profile(series)
    non_missing = series[~missing]
    numeric_success = 0.0
    if len(non_missing) and npf is not None:
        numeric_success = float(pd.to_numeric(non_missing, errors="coerce").notna().mean())
    if (dtype_hint.startswith("int") or dtype_hint.startswith("float") or numeric_success >= .90) and npf is not None:
        prof.update({"kind": "numeric", "numeric": npf})
        return prof
    prof.update({"kind": "categorical", "categorical": categorical_profile(series, top_k, min_count, hash_categories)})
    return prof


def profile_file(data_root: Path, schema_item: dict[str, Any], args, encodings: list[str]):
    rel = Path(schema_item["relative_path"])
    csv_path = data_root / rel
    result = {
        "relative_path": schema_item["relative_path"],
        "filename": schema_item["filename"],
        "columns": schema_item["columns"],
        "dtypes": schema_item.get("dtypes", {}),
        "source_encoding_hint": schema_item.get("encoding_used"),
        "row_count_profiled": 0,
        "columns_profile": {},
        "error": None,
    }
    if not csv_path.exists():
        result["error"] = f"File not found: {csv_path}"
        return result
    df, enc, err = read_csv_safe(csv_path, schema_item.get("encoding_used"), args.sample_rows, encodings)
    if df is None:
        result["error"] = err
        return result
    result["encoding_used"] = enc
    result["row_count_profiled"] = int(len(df))
    for col in schema_item["columns"]:
        result["columns_profile"][col] = profile_column(
            df, col, schema_item.get("dtypes", {}).get(col, "object"),
            args.top_k, args.min_count, args.hash_categories
        )
    return result


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    schema = load_json(Path(args.schema))
    encodings = [e.strip() for e in args.encoding_fallback.split(",") if e.strip()]
    profiles, errors = [], 0
    for i, item in enumerate(schema["csv_schemas"], 1):
        prof = profile_file(data_root, item, args, encodings)
        profiles.append(prof)
        if prof["error"]:
            errors += 1
            print(f"[ERROR] {i}/{len(schema['csv_schemas'])} {item['relative_path']}: {prof['error']}")
        else:
            print(f"[OK] {i}/{len(schema['csv_schemas'])} {item['relative_path']} ({prof['row_count_profiled']} rows)")
    output = {
        "profile_version": 1,
        "source_schema": str(args.schema),
        "data_root": str(data_root),
        "csv_file_count": len(schema["csv_schemas"]),
        "privacy": {
            "raw_rows_exported": False,
            "top_k": args.top_k,
            "min_count": args.min_count,
            "hash_categories": args.hash_categories,
            "sample_rows": args.sample_rows,
        },
        "file_profiles": profiles,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nDistribution profiling completed. Output: {out}; Errors: {errors}")


if __name__ == "__main__":
    main()
