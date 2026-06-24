#!/usr/bin/env python3
"""
Generate synthetic CSV data from dataset_schema.json + aggregate distribution profile.
"""
from __future__ import annotations
import argparse, json, random, re, shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

FALLBACK_CATEGORIES = {
    "SEX": ["M", "F", "U"], "GENDER": ["M", "F"],
    "STATE": ["CA", "TX", "NY", "FL", "PA", "IL", "OH", "GA", "NC", "MI"],
    "ROLE_COD": ["PS", "SS", "C", "I"],
    "OUTC_COD": ["DE", "LT", "HO", "DS", "CA", "RI", "OT"],
    "RPSR_COD": ["CSM", "HP", "UF", "CR", "DT", "CN"],
    "VAX_TYPE": ["COVID19", "FLU3", "HEP", "MMR", "VARCEL", "TDAP"],
    "VAX_ROUTE": ["IM", "SC", "ID", "PO"], "VAX_SITE": ["LA", "RA", "LG", "RG"],
}
TEXT_FALLBACK = [
    "Synthetic record generated from aggregate profile.",
    "No real patient information is included.",
    "Synthetic clinical text placeholder.",
]


def parse_args():
    p = argparse.ArgumentParser(description="Generate profiled synthetic CSV files.")
    p.add_argument("--schema", default="dataset_reports/dataset_schema.json")
    p.add_argument("--profile", default="dataset_reports/dataset_distribution_profile.json")
    p.add_argument("--output-root", default="data/synthetic_profiled")
    p.add_argument("--rows-scale", type=float, default=1.0)
    p.add_argument("--max-rows-per-file", type=int, default=1000)
    p.add_argument("--min-rows-per-file", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--clean", action="store_true")
    return p.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def profile_map(profile: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {p["relative_path"]: p for p in profile.get("file_profiles", [])}


def maybe_missing(v: Any, rate: float):
    return "" if random.random() < rate else v


def sample_numeric(num: dict[str, Any]):
    knots = [
        (.01, num.get("min", 0), num.get("q01", num.get("min", 0))),
        (.04, num.get("q01", 0), num.get("q05", 0)),
        (.20, num.get("q05", 0), num.get("q25", 0)),
        (.25, num.get("q25", 0), num.get("median", 0)),
        (.25, num.get("median", 0), num.get("q75", 0)),
        (.20, num.get("q75", 0), num.get("q95", 0)),
        (.04, num.get("q95", 0), num.get("q99", 0)),
        (.01, num.get("q99", 0), num.get("max", 0)),
    ]
    r, acc = random.random(), 0.0
    lo, hi = num.get("min", 0), num.get("max", 1)
    for p, a, b in knots:
        acc += p
        if r <= acc:
            lo, hi = a, b
            break
    if hi < lo:
        lo, hi = hi, lo
    val = hi if hi == lo else random.uniform(lo, hi)
    return int(round(val)) if num.get("integer_like", False) else round(float(val), 4)


def parse_dt(s: str):
    for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%m/%d/%Y", "%Y%m%d"]:
        try:
            return datetime.strptime(str(s), fmt)
        except Exception:
            pass
    try:
        return pd.to_datetime(s).to_pydatetime()
    except Exception:
        return None


def sample_date(dp: dict[str, Any]):
    lo, hi = parse_dt(dp.get("min", "")), parse_dt(dp.get("max", ""))
    if lo is None or hi is None or hi < lo:
        lo, hi = datetime(2004, 1, 1), datetime(2012, 12, 31)
    dt = lo + timedelta(seconds=random.randint(0, max(int((hi - lo).total_seconds()), 1)))
    fmt = dp.get("format", "infer")
    if fmt == "%Y%m%d": return dt.strftime("%Y%m%d")
    if fmt == "%m/%d/%Y": return dt.strftime("%m/%d/%Y")
    if fmt == "%Y-%m-%d": return dt.strftime("%Y-%m-%d")
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def sample_categorical(col: str, cat: dict[str, Any]) -> str:
    tops = cat.get("top_values", [])
    if tops and not cat.get("hashed", False):
        values = [x["value"] for x in tops]
        probs = [max(float(x.get("probability", 0)), 0) for x in tops]
        total, other = sum(probs), max(float(cat.get("other_probability", 0)), 0)
        if total > 0 and random.random() < total / (total + other):
            probs = [p / total for p in probs]
            return str(np.random.choice(values, p=probs))
    c = col.upper()
    if c in FALLBACK_CATEGORIES:
        return random.choice(FALLBACK_CATEGORIES[c])
    if "TEXT" in c or "NOTE" in c:
        return random.choice(TEXT_FALLBACK)
    median_len = int(cat.get("string_length", {}).get("median", 8) or 8)
    median_len = max(4, min(median_len, 40))
    return f"SYN_{c[:12]}_{random.randint(1, 10**6):0{min(median_len, 12)}d}"


def sample_value(col: str, cp: dict[str, Any], i: int):
    miss = float(cp.get("missing_rate", 0))
    kind = cp.get("kind")
    if kind == "numeric": return maybe_missing(sample_numeric(cp["numeric"]), miss)
    if kind == "date": return maybe_missing(sample_date(cp["date"]), miss)
    if kind == "categorical": return maybe_missing(sample_categorical(col, cp["categorical"]), miss)
    return maybe_missing(f"SYN_{col}_{i}", miss)


def choose_rows(fp: dict[str, Any] | None, args):
    if fp is None:
        return args.min_rows_per_file
    rows = int(round(int(fp.get("row_count_profiled", args.min_rows_per_file)) * args.rows_scale))
    return min(max(args.min_rows_per_file, rows), args.max_rows_per_file)


def extract_year(rel: str):
    m = re.search(r"(20\d{2})", rel)
    return int(m.group(1)) if m else None


def extract_quarter(rel: str):
    m = re.search(r"(20\d{2}Q[1-4])", rel)
    return m.group(1) if m else None


def apply_relationship_ids(df: pd.DataFrame, rel: str):
    n = len(df)
    up = rel.upper()
    if "VAERS" in up and "VAERS_ID" in df.columns:
        year = extract_year(rel) or 2004
        df["VAERS_ID"] = list(range(year * 100000 + 1, year * 100000 + n + 1))
    if "ORDER" in df.columns:
        df["ORDER"] = [(i % 5) + 1 for i in range(n)]
    if "2004Q1_2012Q4" in rel:
        q = extract_quarter(rel) or "2004Q1"
        start = int(q[:4]) * 1000000 + int(q[-1]) * 100000
        if "PRIMARYID" in df.columns: df["PRIMARYID"] = [start + 1 + i for i in range(n)]
        if "CASEID" in df.columns: df["CASEID"] = [start + 50001 + i for i in range(n)]
    if "mimic-iii-clinical-database-demo-1.4" in rel:
        if "row_id" in df.columns: df["row_id"] = list(range(1, n + 1))
        if "subject_id" in df.columns: df["subject_id"] = [10001 + (i % n) for i in range(n)]
        if "hadm_id" in df.columns: df["hadm_id"] = [200001 + (i % n) for i in range(n)]
        if "icustay_id" in df.columns: df["icustay_id"] = [300001 + (i % n) for i in range(n)]
    return df


def generate_file(schema_item: dict[str, Any], fp: dict[str, Any] | None, args):
    cols = schema_item["columns"]
    cps = fp.get("columns_profile", {}) if fp else {}
    rows = choose_rows(fp, args)
    records = []
    for i in range(rows):
        row = {}
        for col in cols:
            row[col] = sample_value(col, cps[col], i) if col in cps else f"SYN_{col}_{i + 1}"
        records.append(row)
    df = pd.DataFrame(records, columns=cols)
    return apply_relationship_ids(df, schema_item["relative_path"])


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    schema = load_json(Path(args.schema))
    pmap = profile_map(load_json(Path(args.profile)))
    out_root = Path(args.output_root)
    if args.clean and out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    generated, missing = 0, 0
    for item in schema["csv_schemas"]:
        rel = item["relative_path"]
        rp = Path(rel)
        if rp.is_absolute() or ".." in rp.parts:
            print(f"[SKIP] Unsafe path: {rel}")
            continue
        fp = pmap.get(rel)
        if fp is None:
            missing += 1
        df = generate_file(item, fp, args)
        out = out_root / rp
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False, encoding="utf-8")
        generated += 1
        print(f"[OK] {out} ({len(df)} rows, {len(df.columns)} columns)")
    print(f"\nProfiled synthetic generation completed. Generated: {generated}; missing profiles: {missing}; output: {out_root}")


if __name__ == "__main__":
    main()
