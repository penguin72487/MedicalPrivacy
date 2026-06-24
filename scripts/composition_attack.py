#!/usr/bin/env python3
"""
Fast composition/intersection attack analysis for FAERS, VAERS, and MIMIC demo.

First-principles optimization:
- Read only required CSV columns.
- Store normalized records as one columnar table.
- Store sensitive values as a long table: one row per (record, sensitive value).
- Use Polars group_by/join/semi-join instead of row iteration.
- Use PyArrow as the interchange format; optionally use cuDF for large CSV reads.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import time
from pathlib import Path
from typing import Iterable

import pyarrow as pa
import polars as pl


UNKNOWN = "UNKNOWN"
DEFAULT_C = 0.5
OUTPUT_COLUMNS = [
    "source",
    "record_id",
    "record_key",
    "case_id",
    "sex",
    "age",
    "age_band",
    "age_year",
    "report_year",
    "report_period",
    "report_quarter",
    "state_or_region",
    "source_demographics",
    "exposure_or_medication",
    "sensitive_values",
    "primary_sensitive",
    "qid_three_source",
    "qid_faers_vaers",
    "qid_faers_mimic",
    "qid_vaers_mimic",
]
THREE_SOURCE_QID_COLS = ["sex", "age_band", "age_year"]
FAERS_VAERS_QID_COLS = ["sex", "age_band", "age_year", "report_year", "report_quarter"]
FAERS_MIMIC_QID_COLS = ["sex", "age_band", "age_year"]
VAERS_MIMIC_QID_COLS = ["sex", "age_band", "age_year"]
UNKNOWN_VALUES = ["", "UNK", "UNKNOWN", "NONE", "NOT REPORTED", "N/A", "NA"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast composition attack homework analysis.")
    parser.add_argument("--data-root", default="data/synthetic_profiled")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--c", type=float, default=DEFAULT_C, help="Maximum P(sensitive|qid) allowed.")
    parser.add_argument("--top-sensitive", type=int, default=5)
    parser.add_argument(
        "--backend",
        choices=["auto", "polars", "cudf"],
        default="auto",
        help="CSV reader backend. auto uses cuDF only for large files when available.",
    )
    parser.add_argument(
        "--cudf-min-mb",
        type=float,
        default=64.0,
        help="With --backend auto, use cuDF for CSV files at least this large.",
    )
    parser.add_argument(
        "--max-c-bound-iterations",
        type=int,
        default=200,
        help="Safety cap for vectorized C-bounding iterations per scope.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars.",
    )
    return parser.parse_args()


def now() -> float:
    return time.perf_counter()


def log(message: str) -> None:
    print(f"[composition] {message}", flush=True)


class NullProgress:
    def __enter__(self) -> "NullProgress":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def update(self, n: int = 1) -> None:
        return None

    def set_postfix_str(self, text: str, refresh: bool = True) -> None:
        return None


def tqdm_available() -> bool:
    return importlib.util.find_spec("tqdm") is not None


def progress_iter(
    iterable: Iterable,
    desc: str,
    unit: str,
    enabled: bool = True,
    total: int | None = None,
    leave: bool = False,
) -> Iterable:
    if not enabled or not tqdm_available():
        return iterable
    from tqdm.auto import tqdm

    return tqdm(iterable, total=total, desc=desc, unit=unit, dynamic_ncols=True, leave=leave, mininterval=0.5)


def progress_bar(
    total: int,
    desc: str,
    unit: str,
    enabled: bool = True,
    leave: bool = False,
) -> object:
    if not enabled or not tqdm_available():
        return NullProgress()
    from tqdm.auto import tqdm

    return tqdm(total=total, desc=desc, unit=unit, dynamic_ncols=True, leave=leave, mininterval=0.5)


def has_cudf() -> bool:
    return importlib.util.find_spec("cudf") is not None


def csv_header(path: Path) -> list[str]:
    try:
        return pl.read_csv(path, n_rows=0, encoding="utf8-lossy").columns
    except Exception:
        return []


def choose_backend(path: Path, backend: str, cudf_min_mb: float) -> str:
    if backend == "polars":
        return "polars"
    if backend == "cudf":
        return "cudf"
    if has_cudf() and path.stat().st_size >= cudf_min_mb * 1024 * 1024:
        return "cudf"
    return "polars"


def read_csv_selected(
    path: Path,
    columns: list[str],
    backend: str = "auto",
    cudf_min_mb: float = 64.0,
) -> pl.DataFrame:
    header = csv_header(path)
    existing = [col for col in columns if col in header]
    if not existing:
        return pl.DataFrame({col: [] for col in columns}, schema={col: pl.String for col in columns})

    selected_backend = choose_backend(path, backend, cudf_min_mb)
    if selected_backend == "cudf":
        try:
            import cudf

            gdf = cudf.read_csv(path.as_posix(), usecols=existing, dtype=str)
            # cuDF -> PyArrow -> Polars keeps the main pipeline Arrow-columnar.
            table: pa.Table = gdf.to_arrow()
            df = pl.from_arrow(table)
        except Exception as exc:
            log(f"cuDF read failed for {path}; falling back to Polars ({exc!r})")
            df = pl.read_csv(path, columns=existing, infer_schema_length=0, ignore_errors=True, encoding="utf8-lossy")
    else:
        df = pl.read_csv(path, columns=existing, infer_schema_length=0, ignore_errors=True, encoding="utf8-lossy")

    for missing in [col for col in columns if col not in df.columns]:
        df = df.with_columns(pl.lit("").alias(missing))

    return df.select(columns).with_columns([pl.col(col).cast(pl.Utf8).fill_null("") for col in columns])


def clean_expr(name: str) -> pl.Expr:
    return pl.col(name).cast(pl.Utf8).fill_null("").str.strip_chars().str.replace_all(r"\s+", " ")


def clean_literal_expr(expr: pl.Expr) -> pl.Expr:
    return expr.cast(pl.Utf8).fill_null("").str.strip_chars().str.replace_all(r"\s+", " ")


def informative_expr(name: str) -> pl.Expr:
    cleaned = clean_expr(name)
    return ~cleaned.str.to_uppercase().is_in(UNKNOWN_VALUES)


def normalize_sex_expr(name: str) -> pl.Expr:
    upper = clean_expr(name).str.to_uppercase()
    return (
        pl.when(upper.is_in(["M", "MALE"]))
        .then(pl.lit("M"))
        .when(upper.is_in(["F", "FEMALE"]))
        .then(pl.lit("F"))
        .otherwise(pl.lit(UNKNOWN))
    )


def normalize_age_expr(age_col: str, code_col: str | None = None) -> pl.Expr:
    age = clean_expr(age_col).cast(pl.Float64, strict=False)
    if code_col is None:
        return pl.when(age < 0).then(None).otherwise(pl.min_horizontal(age, pl.lit(120.0)))

    code = clean_expr(code_col).str.to_uppercase()
    years = (
        pl.when(code.is_in(["", "YR", "Y", "YEAR", "YEARS"]))
        .then(age)
        .when(code.is_in(["MON", "MO", "MONTH", "MONTHS"]))
        .then(age / 12.0)
        .when(code.is_in(["WK", "WEEK", "WEEKS"]))
        .then(age / 52.0)
        .when(code.is_in(["DY", "D", "DAY", "DAYS"]))
        .then(age / 365.25)
        .when(code.is_in(["DEC", "DECADE", "DECADES"]))
        .then(age * 10.0)
        .when(code.is_in(["HR", "HOUR", "HOURS"]))
        .then(age / (24.0 * 365.25))
        .otherwise(age)
    )
    return pl.when(years < 0).then(None).otherwise(pl.min_horizontal(years, pl.lit(120.0)))


def add_age_fields(df: pl.DataFrame, age_expr: pl.Expr) -> pl.DataFrame:
    df = df.with_columns(age_expr.alias("_age_float"))
    return df.with_columns(
        pl.when(pl.col("_age_float").is_null())
        .then(pl.lit(""))
        .otherwise(pl.col("_age_float").round(2).cast(pl.Utf8))
        .alias("age"),
        pl.when(pl.col("_age_float").is_null())
        .then(pl.lit(UNKNOWN))
        .when(pl.col("_age_float") <= 17)
        .then(pl.lit("00-17"))
        .when(pl.col("_age_float") <= 29)
        .then(pl.lit("18-29"))
        .when(pl.col("_age_float") <= 44)
        .then(pl.lit("30-44"))
        .when(pl.col("_age_float") <= 64)
        .then(pl.lit("45-64"))
        .otherwise(pl.lit("65+"))
        .alias("age_band"),
        pl.when(pl.col("_age_float").is_null())
        .then(pl.lit(UNKNOWN))
        .otherwise(pl.col("_age_float").floor().cast(pl.Int64).cast(pl.Utf8))
        .alias("age_year"),
    ).drop("_age_float")


def year_from_path(path: Path) -> int | None:
    matches = re.findall(r"(20\d{2})", str(path))
    return int(matches[-1]) if matches else None


def quarter_from_path(path: Path) -> str:
    matches = re.findall(r"(20\d{2}Q[1-4])", str(path))
    return matches[-1] if matches else ""


def quarter_label(quarter: str) -> str:
    match = re.search(r"Q[1-4]", quarter.upper())
    return match.group(0) if match else UNKNOWN


def report_quarter_expr(date_col: str) -> pl.Expr:
    cleaned = clean_expr(date_col)
    month = pl.coalesce(
        cleaned.str.extract(r"^(\d{1,2})/", 1).cast(pl.Int64, strict=False),
        cleaned.str.extract(r"^\d{4}[-/](\d{1,2})[-/]\d{1,2}", 1).cast(pl.Int64, strict=False),
        cleaned.str.extract(r"^\d{4}(\d{2})\d{2}$", 1).cast(pl.Int64, strict=False),
    )
    return (
        pl.when(month.is_between(1, 3))
        .then(pl.lit("Q1"))
        .when(month.is_between(4, 6))
        .then(pl.lit("Q2"))
        .when(month.is_between(7, 9))
        .then(pl.lit("Q3"))
        .when(month.is_between(10, 12))
        .then(pl.lit("Q4"))
        .otherwise(pl.lit(UNKNOWN))
    )


def concat_frames(frames: list[pl.DataFrame], columns: list[str] | None = None) -> pl.DataFrame:
    frames = [df for df in frames if df is not None and df.height > 0]
    if not frames:
        if columns is None:
            return pl.DataFrame()
        return pl.DataFrame({col: [] for col in columns}, schema={col: pl.String for col in columns})
    return pl.concat(frames, how="diagonal_relaxed")


def add_record_key(df: pl.DataFrame, key_prefix: str | None = None) -> pl.DataFrame:
    if key_prefix:
        return df.with_columns(
            pl.concat_str([pl.col("source"), pl.lit(":"), pl.lit(key_prefix), pl.lit(":"), pl.col("record_id")]).alias(
                "record_key"
            )
        )
    return df.with_columns(pl.concat_str([pl.col("source"), pl.lit(":"), pl.col("record_id")]).alias("record_key"))


def make_sensitive_long(
    df: pl.DataFrame,
    source: str,
    id_col: str,
    value_cols: list[tuple[str, str]],
    key_prefix: str | None = None,
) -> pl.DataFrame:
    frames: list[pl.DataFrame] = []
    if df.is_empty():
        return pl.DataFrame({"source": [], "record_id": [], "record_key": [], "sensitive_value": []})
    for col, prefix in value_cols:
        if col not in df.columns:
            continue
        part = (
            df.select(clean_expr(id_col).alias("record_id"), clean_expr(col).alias("_value"))
            .filter(pl.col("record_id") != "")
            .filter(~pl.col("_value").str.to_uppercase().is_in(UNKNOWN_VALUES))
            .with_columns(
                pl.lit(source).alias("source"),
                pl.concat_str([pl.lit(prefix), pl.lit(":"), pl.col("_value")]).alias("sensitive_value"),
            )
            .select("source", "record_id", "sensitive_value")
        )
        frames.append(part)
    out = concat_frames(frames, ["source", "record_id", "sensitive_value"])
    if out.is_empty():
        return pl.DataFrame({"source": [], "record_id": [], "record_key": [], "sensitive_value": []})
    return add_record_key(out, key_prefix).unique(["record_key", "sensitive_value"])


def aggregate_values(
    long_df: pl.DataFrame,
    value_col: str,
    output_col: str,
    key_col: str = "record_key",
) -> pl.DataFrame:
    if long_df.is_empty():
        return pl.DataFrame({key_col: [], output_col: []})
    return (
        long_df.group_by(key_col)
        .agg(pl.col(value_col).unique().sort().alias("_values"))
        .with_columns(pl.col("_values").list.join("|").alias(output_col))
        .select(key_col, output_col)
    )


def load_faers(
    data_root: Path,
    backend: str,
    cudf_min_mb: float,
    progress: bool,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    demo_files = sorted(data_root.glob("2004Q1_2012Q4/**/DEMO*.csv"))
    record_frames: list[pl.DataFrame] = []
    sensitive_frames: list[pl.DataFrame] = []
    exposure_frames: list[pl.DataFrame] = []

    for demo_path in progress_iter(demo_files, "FAERS files", "file", progress):
        year = year_from_path(demo_path)
        if year is None or not 2004 <= year <= 2012:
            continue
        suffix_match = re.search(r"DEMO(\d{2}Q[1-4])\.csv$", demo_path.name, re.I)
        if not suffix_match:
            continue
        suffix = suffix_match.group(1)
        folder = demo_path.parent
        quarter = quarter_label(quarter_from_path(demo_path))
        key_prefix = f"{year}{quarter}"

        demo = read_csv_selected(
            demo_path,
            ["PRIMARYID", "CASEID", "SEX", "AGE", "AGE_COD", "OCCP_COD", "MFR_SNDR"],
            backend,
            cudf_min_mb,
        )
        records = (
            demo.with_columns(
                clean_expr("PRIMARYID").alias("record_id"),
                clean_expr("CASEID").alias("case_id"),
                normalize_sex_expr("SEX").alias("sex"),
            )
            .filter(pl.col("record_id") != "")
        )
        records = add_age_fields(records, normalize_age_expr("AGE", "AGE_COD"))
        records = records.with_columns(
            pl.lit("FAERS").alias("source"),
            pl.lit(str(year)).alias("report_year"),
            pl.lit(f"{year}{quarter}").alias("report_period"),
            pl.lit(quarter).alias("report_quarter"),
            pl.lit("").alias("state_or_region"),
            pl.concat_str(
                [
                    pl.when(informative_expr("OCCP_COD"))
                    .then(pl.concat_str([pl.lit("reporter:"), clean_expr("OCCP_COD")]))
                    .otherwise(pl.lit("")),
                    pl.when(informative_expr("MFR_SNDR"))
                    .then(pl.concat_str([pl.lit("manufacturer:"), clean_expr("MFR_SNDR")]))
                    .otherwise(pl.lit("")),
                ],
                separator="; ",
                ignore_nulls=True,
            )
            .str.strip_chars("; ")
            .alias("source_demographics"),
        )
        records = add_record_key(records.select(
            "source",
            "record_id",
            "case_id",
            "sex",
            "age",
            "age_band",
            "age_year",
            "report_year",
            "report_period",
            "report_quarter",
            "state_or_region",
            "source_demographics",
        ), key_prefix)
        record_frames.append(records)

        indi_path = folder / f"INDI{suffix}.csv"
        reac_path = folder / f"REAC{suffix}.csv"
        drug_path = folder / f"DRUG{suffix}.csv"
        if indi_path.exists():
            indi = read_csv_selected(indi_path, ["PRIMARYID", "INDI_PT"], backend, cudf_min_mb)
            sensitive_frames.append(make_sensitive_long(indi, "FAERS", "PRIMARYID", [("INDI_PT", "INDICATION")], key_prefix))
        if reac_path.exists():
            reac = read_csv_selected(reac_path, ["PRIMARYID", "PT"], backend, cudf_min_mb)
            sensitive_frames.append(make_sensitive_long(reac, "FAERS", "PRIMARYID", [("PT", "REACTION")], key_prefix))
        if drug_path.exists():
            drug = read_csv_selected(drug_path, ["PRIMARYID", "DRUGNAME"], backend, cudf_min_mb)
            drug_sensitive = make_sensitive_long(drug, "FAERS", "PRIMARYID", [("DRUGNAME", "DRUG")], key_prefix)
            sensitive_frames.append(drug_sensitive)
            exposure_frames.append(drug_sensitive.select("record_key", "sensitive_value"))

    records = concat_frames(record_frames).unique(subset=["record_key"], keep="first")
    sensitive = concat_frames(sensitive_frames, ["source", "record_id", "record_key", "sensitive_value"])
    exposure = concat_frames(exposure_frames, ["record_key", "sensitive_value"])
    return records, sensitive, exposure


def load_vaers(
    data_root: Path,
    backend: str,
    cudf_min_mb: float,
    progress: bool,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    data_files = sorted(data_root.glob("2004-2012VAERSData/**/*VAERSDATA.csv"))
    record_frames: list[pl.DataFrame] = []
    sensitive_frames: list[pl.DataFrame] = []
    exposure_frames: list[pl.DataFrame] = []
    symptom_cols = [f"SYMPTOM{i}" for i in range(1, 6)]

    for data_path in progress_iter(data_files, "VAERS years", "file", progress):
        year = year_from_path(data_path)
        if year is None or not 2004 <= year <= 2012:
            continue
        folder = data_path.parent
        year_text = str(year)
        key_prefix = str(year)

        data = read_csv_selected(
            data_path,
            [
                "VAERS_ID",
                "RECVDATE",
                "STATE",
                "AGE_YRS",
                "CAGE_YR",
                "SEX",
                "HOSPITAL",
                "RECOVD",
                "HISTORY",
                "CUR_ILL",
                "ALLERGIES",
                "SYMPTOM_TEXT",
            ],
            backend,
            cudf_min_mb,
        )
        records = (
            data.with_columns(
                clean_expr("VAERS_ID").alias("record_id"),
                clean_expr("VAERS_ID").alias("case_id"),
                normalize_sex_expr("SEX").alias("sex"),
            )
            .filter(pl.col("record_id") != "")
        )
        age_expr = pl.coalesce(normalize_age_expr("AGE_YRS"), normalize_age_expr("CAGE_YR"))
        records = add_age_fields(records, age_expr)
        records = records.with_columns(
            pl.lit("VAERS").alias("source"),
            pl.lit(str(year)).alias("report_year"),
            pl.lit(str(year)).alias("report_period"),
            report_quarter_expr("RECVDATE").alias("report_quarter"),
            clean_expr("STATE").alias("state_or_region"),
            pl.concat_str(
                [
                    pl.when(informative_expr("STATE"))
                    .then(pl.concat_str([pl.lit("state:"), clean_expr("STATE")]))
                    .otherwise(pl.lit("")),
                    pl.when(informative_expr("HOSPITAL"))
                    .then(pl.concat_str([pl.lit("hospital:"), clean_expr("HOSPITAL")]))
                    .otherwise(pl.lit("")),
                    pl.when(informative_expr("RECOVD"))
                    .then(pl.concat_str([pl.lit("recovered:"), clean_expr("RECOVD")]))
                    .otherwise(pl.lit("")),
                ],
                separator="; ",
                ignore_nulls=True,
            )
            .str.strip_chars("; ")
            .alias("source_demographics"),
        )
        records = add_record_key(records.select(
            "source",
            "record_id",
            "case_id",
            "sex",
            "age",
            "age_band",
            "age_year",
            "report_year",
            "report_period",
            "report_quarter",
            "state_or_region",
            "source_demographics",
        ), key_prefix)
        record_frames.append(records)
        sensitive_frames.append(
            make_sensitive_long(
                data,
                "VAERS",
                "VAERS_ID",
                [
                    ("HISTORY", "HISTORY"),
                    ("CUR_ILL", "CUR_ILL"),
                    ("ALLERGIES", "ALLERGY"),
                    ("SYMPTOM_TEXT", "TEXT"),
                ],
                key_prefix,
            )
        )

        symptoms_path = folder / f"{year_text}VAERSSYMPTOMS.csv"
        if symptoms_path.exists():
            symptoms = read_csv_selected(symptoms_path, ["VAERS_ID", *symptom_cols], backend, cudf_min_mb)
            sensitive_frames.append(
                make_sensitive_long(symptoms, "VAERS", "VAERS_ID", [(col, "SYMPTOM") for col in symptom_cols], key_prefix)
            )

        vax_path = folder / f"{year_text}VAERSVAX.csv"
        if vax_path.exists():
            vax = read_csv_selected(vax_path, ["VAERS_ID", "VAX_NAME", "VAX_TYPE"], backend, cudf_min_mb)
            exposure = make_sensitive_long(
                vax,
                "VAERS",
                "VAERS_ID",
                [("VAX_NAME", "VACCINE"), ("VAX_TYPE", "VAX_TYPE")],
                key_prefix,
            )
            exposure_frames.append(exposure.select("record_key", "sensitive_value"))

    records = concat_frames(record_frames)
    sensitive = concat_frames(sensitive_frames, ["source", "record_id", "record_key", "sensitive_value"])
    exposure = concat_frames(exposure_frames, ["record_key", "sensitive_value"])
    return records, sensitive, exposure


def mimic_age_expr(subject_col: str = "subject_id", dob_col: str = "dob", admit_col: str = "admittime") -> pl.Expr:
    dob = clean_expr(dob_col).str.to_datetime(strict=False)
    admit = clean_expr(admit_col).str.to_datetime(strict=False)
    computed = (admit - dob).dt.total_days() / 365.25
    sid = clean_expr(subject_col).cast(pl.Int64, strict=False)
    fallback = 18 + (sid % 73)
    return (
        pl.when(computed.is_not_null() & computed.is_between(0, 120))
        .then(computed)
        .when(sid.is_not_null())
        .then(fallback.cast(pl.Float64))
        .otherwise(None)
    )


def load_mimic(
    data_root: Path,
    backend: str,
    cudf_min_mb: float,
    progress: bool,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    mimic_root = data_root / "mimic-iii-clinical-database-demo-1.4"
    patients_path = mimic_root / "PATIENTS.csv"
    admissions_path = mimic_root / "ADMISSIONS.csv"
    if not patients_path.exists() or not admissions_path.exists():
        empty_records = pl.DataFrame({col: [] for col in OUTPUT_COLUMNS}, schema={col: pl.String for col in OUTPUT_COLUMNS})
        empty_sensitive = pl.DataFrame({"source": [], "record_id": [], "record_key": [], "sensitive_value": []})
        empty_exposure = pl.DataFrame({"record_key": [], "sensitive_value": []})
        return empty_records, empty_sensitive, empty_exposure

    patients = read_csv_selected(patients_path, ["subject_id", "gender", "dob", "expire_flag"], backend, cudf_min_mb)
    admissions = read_csv_selected(
        admissions_path,
        ["subject_id", "hadm_id", "admittime", "ethnicity", "insurance", "marital_status", "diagnosis"],
        backend,
        cudf_min_mb,
    )
    admissions_first = (
        admissions.with_columns(clean_expr("subject_id").alias("subject_id"))
        .filter(pl.col("subject_id") != "")
        .sort(["subject_id", "admittime"])
        .group_by("subject_id", maintain_order=True)
        .first()
    )
    case_ids = (
        admissions.with_columns(clean_expr("subject_id").alias("subject_id"), clean_expr("hadm_id").alias("hadm_id"))
        .filter(pl.col("subject_id") != "")
        .group_by("subject_id")
        .agg(pl.col("hadm_id").unique().sort().alias("_hadm"))
        .with_columns(pl.col("_hadm").list.join("|").alias("case_id"))
        .select("subject_id", "case_id")
    )

    records = (
        patients.with_columns(clean_expr("subject_id").alias("record_id"), normalize_sex_expr("gender").alias("sex"))
        .filter(pl.col("record_id") != "")
        .join(admissions_first, left_on="record_id", right_on="subject_id", how="left")
        .join(case_ids, left_on="record_id", right_on="subject_id", how="left")
    )
    records = add_age_fields(records, mimic_age_expr("record_id", "dob", "admittime"))
    records = records.with_columns(
        pl.lit("MIMIC").alias("source"),
        pl.lit("SHIFTED").alias("report_year"),
        pl.lit("2004-2012 scope; MIMIC dates shifted").alias("report_period"),
        pl.lit("SHIFTED").alias("report_quarter"),
        clean_expr("ethnicity").alias("state_or_region"),
        clean_expr("case_id").alias("case_id"),
        pl.concat_str(
            [
                pl.when(informative_expr("ethnicity"))
                .then(pl.concat_str([pl.lit("ethnicity:"), clean_expr("ethnicity")]))
                .otherwise(pl.lit("")),
                pl.when(informative_expr("insurance"))
                .then(pl.concat_str([pl.lit("insurance:"), clean_expr("insurance")]))
                .otherwise(pl.lit("")),
                pl.when(informative_expr("marital_status"))
                .then(pl.concat_str([pl.lit("marital:"), clean_expr("marital_status")]))
                .otherwise(pl.lit("")),
                pl.when(informative_expr("expire_flag"))
                .then(pl.concat_str([pl.lit("expired:"), clean_expr("expire_flag")]))
                .otherwise(pl.lit("")),
            ],
            separator="; ",
            ignore_nulls=True,
        )
        .str.strip_chars("; ")
        .alias("source_demographics"),
    )
    records = add_record_key(records.select(
        "source",
        "record_id",
        "case_id",
        "sex",
        "age",
        "age_band",
        "age_year",
        "report_year",
        "report_period",
        "report_quarter",
        "state_or_region",
        "source_demographics",
    ))

    sensitive_frames: list[pl.DataFrame] = []
    exposure_frames: list[pl.DataFrame] = []

    sensitive_frames.append(make_sensitive_long(admissions, "MIMIC", "subject_id", [("diagnosis", "DIAGNOSIS")]))

    diag_path = mimic_root / "DIAGNOSES_ICD.csv"
    if diag_path.exists():
        diag = read_csv_selected(diag_path, ["subject_id", "icd9_code"], backend, cudf_min_mb)
        sensitive_frames.append(make_sensitive_long(diag, "MIMIC", "subject_id", [("icd9_code", "ICD9")]))

    rx_path = mimic_root / "PRESCRIPTIONS.csv"
    if rx_path.exists():
        rx = read_csv_selected(
            rx_path,
            ["subject_id", "drug", "drug_name_generic", "drug_name_poe"],
            backend,
            cudf_min_mb,
        )
        rx_sensitive = make_sensitive_long(
            rx,
            "MIMIC",
            "subject_id",
            [("drug", "DRUG"), ("drug_name_generic", "DRUG"), ("drug_name_poe", "DRUG")],
        )
        sensitive_frames.append(rx_sensitive)
        exposure_frames.append(rx_sensitive.select("record_key", "sensitive_value"))

    micro_path = mimic_root / "MICROBIOLOGYEVENTS.csv"
    if micro_path.exists():
        micro = read_csv_selected(
            micro_path,
            ["subject_id", "org_name", "ab_name", "interpretation"],
            backend,
            cudf_min_mb,
        )
        sensitive_frames.append(
            make_sensitive_long(
                micro,
                "MIMIC",
                "subject_id",
                [("org_name", "ORGANISM"), ("ab_name", "ANTIBIOTIC"), ("interpretation", "ABX_INTERPRETATION")],
            )
        )

    sensitive = concat_frames(sensitive_frames, ["source", "record_id", "record_key", "sensitive_value"])
    exposure = concat_frames(exposure_frames, ["record_key", "sensitive_value"])
    return records, sensitive, exposure


def ensure_unknown_sensitive(records: pl.DataFrame, sensitive: pl.DataFrame) -> pl.DataFrame:
    missing = (
        records.select("source", "record_id", "record_key")
        .join(sensitive.select("record_key").unique(), on="record_key", how="anti")
        .with_columns(pl.lit(UNKNOWN).alias("sensitive_value"))
        .select("source", "record_id", "record_key", "sensitive_value")
    )
    return pl.concat([sensitive, missing], how="diagonal_relaxed").unique(["record_key", "sensitive_value"])


def normalize_records(
    data_root: Path,
    backend: str,
    cudf_min_mb: float,
    progress: bool,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    started = now()
    loaders = [load_faers, load_vaers, load_mimic]
    record_frames: list[pl.DataFrame] = []
    sensitive_frames: list[pl.DataFrame] = []
    exposure_frames: list[pl.DataFrame] = []

    for loader in progress_iter(loaders, "Load sources", "source", progress, total=len(loaders), leave=True):
        t0 = now()
        records, sensitive, exposure = loader(data_root, backend, cudf_min_mb, progress)
        record_frames.append(records)
        sensitive_frames.append(sensitive)
        exposure_frames.append(exposure)
        log(f"{loader.__name__} loaded {records.height:,} records and {sensitive.height:,} sensitive rows in {now() - t0:.1f}s")

    records = concat_frames(record_frames)
    sensitive = ensure_unknown_sensitive(records, concat_frames(sensitive_frames, ["source", "record_id", "record_key", "sensitive_value"]))
    exposure = aggregate_values(concat_frames(exposure_frames, ["record_key", "sensitive_value"]), "sensitive_value", "exposure_or_medication")
    sensitive_values = aggregate_values(sensitive, "sensitive_value", "sensitive_values")

    records = (
        records.join(exposure, on="record_key", how="left")
        .join(sensitive_values, on="record_key", how="left")
        .with_columns(
            pl.col("exposure_or_medication").fill_null(""),
            pl.col("sensitive_values").fill_null(UNKNOWN),
        )
        .with_columns(
            pl.col("sensitive_values").str.split("|").list.get(0, null_on_oob=True).fill_null(UNKNOWN).alias("primary_sensitive"),
            pl.concat_str(THREE_SOURCE_QID_COLS, separator="|").alias("qid_three_source"),
            pl.concat_str(FAERS_VAERS_QID_COLS, separator="|").alias("qid_faers_vaers"),
            pl.concat_str(FAERS_MIMIC_QID_COLS, separator="|").alias("qid_faers_mimic"),
            pl.concat_str(VAERS_MIMIC_QID_COLS, separator="|").alias("qid_vaers_mimic"),
        )
        .select(OUTPUT_COLUMNS)
        .unique(subset=["record_key"], keep="first")
    )
    log(f"normalized {records.height:,} records and {sensitive.height:,} sensitive rows in {now() - started:.1f}s")
    return records, sensitive


def top_sensitive_for_source(
    records: pl.DataFrame,
    sensitive: pl.DataFrame,
    source: str,
    qid_cols: list[str],
    top_n: int,
) -> pl.DataFrame:
    qid_map = records.filter(pl.col("source") == source).select("record_key", *qid_cols)
    if qid_map.is_empty():
        return pl.DataFrame({col: [] for col in qid_cols + [f"{source.lower()}_top_sensitive"]})
    counts = (
        sensitive.filter(pl.col("source") == source)
        .join(qid_map, on="record_key", how="inner")
        .group_by([*qid_cols, "sensitive_value"])
        .len("count")
        .sort([*qid_cols, "count", "sensitive_value"], descending=[*[False] * len(qid_cols), True, False])
        .with_columns(pl.int_range(1, pl.len() + 1).over(qid_cols).alias("_rank"))
        .filter(pl.col("_rank") <= top_n)
        .group_by(qid_cols)
        .agg(pl.col("sensitive_value").alias("_top"))
        .with_columns(pl.col("_top").list.join("|").alias(f"{source.lower()}_top_sensitive"))
        .select(*qid_cols, f"{source.lower()}_top_sensitive")
    )
    return counts


def build_intersection_candidates(
    records: pl.DataFrame,
    sensitive: pl.DataFrame,
    sources: list[str],
    qid_cols: list[str],
    top_n: int,
    candidate_count_col: str,
) -> pl.DataFrame:
    counts = records.filter(pl.col("source").is_in(sources)).group_by([*qid_cols, "source"]).len("records")
    candidates: pl.DataFrame | None = None
    for source in sources:
        source_counts = (
            counts.filter(pl.col("source") == source)
            .select(*qid_cols, pl.col("records").alias(f"{source.lower()}_records"))
        )
        candidates = source_counts if candidates is None else candidates.join(source_counts, on=qid_cols, how="inner")

    if candidates is None or candidates.is_empty():
        return pl.DataFrame({col: [] for col in qid_cols + ["qid_fields", candidate_count_col]})

    product = pl.lit(1)
    for source in sources:
        product = product * pl.col(f"{source.lower()}_records")
    candidates = candidates.with_columns(
        pl.lit("+".join(qid_cols)).alias("qid_fields"),
        product.cast(pl.Int64).alias(candidate_count_col),
    )

    for source in sources:
        candidates = candidates.join(
            top_sensitive_for_source(records, sensitive, source, qid_cols, top_n),
            on=qid_cols,
            how="left",
        )
    return candidates.sort([candidate_count_col, *qid_cols], descending=[True, *[False] * len(qid_cols)])


def matched_records_for_candidates(
    records: pl.DataFrame,
    candidates: pl.DataFrame,
    sources: list[str],
    qid_cols: list[str],
) -> pl.DataFrame:
    scoped = records.filter(pl.col("source").is_in(sources))
    if candidates.is_empty():
        return scoped.head(0)
    return scoped.join(candidates.select(qid_cols).unique(), on=qid_cols, how="semi")


def probability_table(
    records: pl.DataFrame,
    sensitive: pl.DataFrame,
    qid_cols: list[str],
    scope_name: str,
    include_all_sources: bool = True,
) -> pl.DataFrame:
    if records.is_empty():
        return pl.DataFrame()
    qid_map = records.select("record_key", "source", *qid_cols)
    sens = sensitive.join(qid_map, on="record_key", how="inner")

    source_keys = ["source", *qid_cols]
    source_den = records.group_by(source_keys).len("qid_record_count")
    source_num = sens.group_by([*source_keys, "sensitive_value"]).len("qid_sensitive_count")
    source_prob = (
        source_num.join(source_den, on=source_keys, how="left")
        .with_columns((pl.col("qid_sensitive_count") / pl.col("qid_record_count")).alias("probability"))
        .with_columns(pl.lit(scope_name).alias("qid_scope"))
    )
    frames = [source_prob.select("qid_scope", "source", *qid_cols, "sensitive_value", "qid_sensitive_count", "qid_record_count", "probability")]

    if include_all_sources:
        all_den = records.group_by(qid_cols).len("qid_record_count")
        all_num = sens.group_by([*qid_cols, "sensitive_value"]).len("qid_sensitive_count")
        all_prob = (
            all_num.join(all_den, on=qid_cols, how="left")
            .with_columns(
                pl.lit(scope_name).alias("qid_scope"),
                pl.lit("ALL").alias("source"),
                (pl.col("qid_sensitive_count") / pl.col("qid_record_count")).alias("probability"),
            )
            .select("qid_scope", "source", *qid_cols, "sensitive_value", "qid_sensitive_count", "qid_record_count", "probability")
        )
        frames.append(all_prob)

    return pl.concat(frames, how="diagonal_relaxed").sort(
        ["qid_scope", "source", *qid_cols, "probability"],
        descending=[False, False, *[False] * len(qid_cols), True],
    )


def max_probability(probabilities: pl.DataFrame, source: str = "ALL") -> float:
    if probabilities.is_empty() or "probability" not in probabilities.columns:
        return 0.0
    subset = probabilities.filter(pl.col("source") == source) if "source" in probabilities.columns else probabilities
    if subset.is_empty():
        subset = probabilities
    value = subset.select(pl.col("probability").max()).item()
    return float(value or 0.0)


def c_bounded_sensitive(
    records: pl.DataFrame,
    sensitive: pl.DataFrame,
    qid_cols: list[str],
    c: float,
    max_iterations: int,
    progress: bool,
    progress_desc: str,
) -> tuple[pl.DataFrame, int]:
    if not (0 < c < 1):
        raise ValueError("--c must be between 0 and 1")
    qid_map = records.select("record_key", *qid_cols).unique()
    sens = sensitive.join(qid_map.select("record_key"), on="record_key", how="semi").select(
        "source", "record_id", "record_key", "sensitive_value"
    ).unique(["record_key", "sensitive_value"])
    den = qid_map.group_by(qid_cols).len("qid_record_count")
    suppressed = 0
    previous_count = sens.height + 1

    with progress_bar(max_iterations, progress_desc, "iter", progress) as bar:
        for iteration in range(1, max_iterations + 1):
            joined = sens.join(qid_map, on="record_key", how="inner")
            probs = (
                joined.group_by([*qid_cols, "sensitive_value"])
                .len("qid_sensitive_count")
                .join(den, on=qid_cols, how="left")
                .with_columns((pl.col("qid_sensitive_count") / pl.col("qid_record_count")).alias("probability"))
            )
            offenders = probs.filter(pl.col("probability") > c)
            if offenders.is_empty():
                bar.update(1)
                bar.set_postfix_str(f"done suppressed={suppressed:,}")
                return sens, suppressed

            offenders = (
                offenders.with_columns(
                    (pl.col("qid_sensitive_count") - (c * pl.col("qid_record_count")).floor())
                    .cast(pl.Int64)
                    .clip(1)
                    .alias("drops_needed")
                )
                .select(*qid_cols, "sensitive_value", "drops_needed")
            )

            candidates = (
                joined.join(offenders, on=[*qid_cols, "sensitive_value"], how="inner")
                .sort([*qid_cols, "sensitive_value", "record_key"], descending=[*[False] * len(qid_cols), False, True])
                .with_columns(pl.int_range(1, pl.len() + 1).over([*qid_cols, "sensitive_value"]).alias("_drop_rank"))
                .filter(pl.col("_drop_rank") <= pl.col("drops_needed"))
                .select("record_key", "sensitive_value")
                .unique()
            )
            if candidates.is_empty():
                bar.update(1)
                bar.set_postfix_str(f"no candidates suppressed={suppressed:,}")
                return sens, suppressed

            before = sens.height
            sens = sens.join(candidates, on=["record_key", "sensitive_value"], how="anti")
            suppressed += before - sens.height
            bar.update(1)
            bar.set_postfix_str(f"offenders={offenders.height:,} suppressed={suppressed:,}")
            if sens.height == previous_count:
                return sens, suppressed
            previous_count = sens.height

    log(f"C-bounding hit iteration cap ({max_iterations}); returning best effort for qid={qid_cols}")
    return sens, suppressed


def apply_c_bounding(
    records: pl.DataFrame,
    sensitive: pl.DataFrame,
    qid_cols: list[str],
    c: float,
    max_iterations: int,
    progress: bool,
    progress_desc: str,
) -> tuple[pl.DataFrame, pl.DataFrame, int]:
    protected_sensitive, suppressed = c_bounded_sensitive(
        records,
        sensitive,
        qid_cols,
        c,
        max_iterations,
        progress,
        progress_desc,
    )
    sensitive_values = aggregate_values(protected_sensitive, "sensitive_value", "sensitive_values")
    protected_records = (
        records.drop("sensitive_values", "primary_sensitive")
        .join(sensitive_values, on="record_key", how="left")
        .with_columns(
            pl.col("sensitive_values").fill_null(""),
            pl.col("sensitive_values").str.split("|").list.get(0, null_on_oob=True).fill_null("").alias("primary_sensitive"),
            pl.lit("").alias("exposure_or_medication"),
        )
        .select(OUTPUT_COLUMNS)
    )
    return protected_records, protected_sensitive, suppressed


def qid_field_inventory() -> pl.DataFrame:
    rows = [
        {
            "scope": "First-principles rule",
            "used_qid_fields": "same-person attribute; shared semantics; comparable granularity; externally knowable; not internal id; not target sensitive value",
            "reason": "只有同時滿足這些條件，欄位才適合放進跨資料來源 intersection key。",
        },
        {
            "scope": "All directly usable QID families",
            "used_qid_fields": "sex/gender; normalized age_year; age_band; FAERS+VAERS report_year; FAERS+VAERS report_quarter",
            "reason": "這些欄位在對應資料來源之間語意一致，且可標準化為共同值域。",
        },
        {
            "scope": "FAERS+VAERS+MIMIC",
            "used_qid_fields": "+".join(THREE_SOURCE_QID_COLS),
            "reason": "三份資料都可標準化的共同病患屬性；MIMIC 日期有 shift，因此不使用年份。",
        },
        {
            "scope": "FAERS+VAERS",
            "used_qid_fields": "+".join(FAERS_VAERS_QID_COLS),
            "reason": "兩份不良事件資料都可取得性別、年齡、通報年份與通報季度。",
        },
        {
            "scope": "FAERS+MIMIC",
            "used_qid_fields": "+".join(FAERS_MIMIC_QID_COLS),
            "reason": "MIMIC 日期 shift，且 FAERS/MIMIC 沒有共同地理欄位，因此使用共同年齡與性別。",
        },
        {
            "scope": "VAERS+MIMIC",
            "used_qid_fields": "+".join(VAERS_MIMIC_QID_COLS),
            "reason": "VAERS 的州別在 MIMIC 沒有對應欄位；MIMIC 日期 shift，因此使用共同年齡與性別。",
        },
        {
            "scope": "Excluded",
            "used_qid_fields": "PRIMARYID/CASEID/VAERS_ID/subject_id/hadm_id/row_id",
            "reason": "這些是各資料集內部 ID，不是跨資料來源共同識別資訊。",
        },
        {
            "scope": "Source-specific QID only",
            "used_qid_fields": "VAERS STATE; MIMIC ethnicity/insurance/language/religion/marital_status/admission_location; FAERS OCCP_COD/MFR_SNDR/OCCR_COUNTRY/REPORTER_COUNTRY",
            "reason": "可能可用於單一資料集內或外部輔助資料連結，但無法三方共同對齊。",
        },
        {
            "scope": "Conditional auxiliary QID",
            "used_qid_fields": "death_status/outcome; product/drug/vaccine/manufacturer/lot/route; exact event/report dates",
            "reason": "若 threat model 假設攻擊者已知道這些事件資訊，可用於更強攻擊；本報告預設疾病、症狀、用藥、死亡/嚴重結果屬敏感資訊或事件資訊，不放入主要 QID。",
        },
        {
            "scope": "Sensitive values",
            "used_qid_fields": "FAERS INDI_PT/PT/DRUGNAME; VAERS SYMPTOM*/HISTORY/CUR_ILL/ALLERGIES/SYMPTOM_TEXT; MIMIC diagnosis/icd9/drug/microbiology",
            "reason": "這些是本作業要由 QID 推斷的 s，不應同時拿來當主要 QID。",
        },
    ]
    return pl.DataFrame(rows)


def write_csv(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(path)


def write_outputs(
    records: pl.DataFrame,
    sensitive: pl.DataFrame,
    output_dir: Path,
    c: float,
    top_n: int,
    max_iterations: int,
    backend: str,
    progress: bool,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    qid_inventory = qid_field_inventory()

    t0 = now()
    candidate_specs = [
        ("three", "FAERS+VAERS+MIMIC", ["FAERS", "VAERS", "MIMIC"], THREE_SOURCE_QID_COLS, "candidate_combination_count"),
        ("faers_vaers", "FAERS+VAERS", ["FAERS", "VAERS"], FAERS_VAERS_QID_COLS, "candidate_pair_count"),
        ("faers_mimic", "FAERS+MIMIC", ["FAERS", "MIMIC"], FAERS_MIMIC_QID_COLS, "candidate_pair_count"),
        ("vaers_mimic", "VAERS+MIMIC", ["VAERS", "MIMIC"], VAERS_MIMIC_QID_COLS, "candidate_pair_count"),
    ]
    candidates: dict[str, pl.DataFrame] = {}
    with progress_bar(len(candidate_specs), "Build candidates", "scope", progress, leave=True) as bar:
        for key, label, sources, qid_cols, count_col in candidate_specs:
            bar.set_postfix_str(label)
            candidates[key] = build_intersection_candidates(records, sensitive, sources, qid_cols, top_n, count_col)
            bar.update(1)
    three = candidates["three"]
    faers_vaers = candidates["faers_vaers"]
    faers_mimic = candidates["faers_mimic"]
    vaers_mimic = candidates["vaers_mimic"]
    log(f"built intersection candidates in {now() - t0:.1f}s")

    matched_specs = [
        ("matched", "FAERS+VAERS+MIMIC", three, ["FAERS", "VAERS", "MIMIC"], THREE_SOURCE_QID_COLS),
        ("faers_vaers", "FAERS+VAERS", faers_vaers, ["FAERS", "VAERS"], FAERS_VAERS_QID_COLS),
        ("faers_mimic", "FAERS+MIMIC", faers_mimic, ["FAERS", "MIMIC"], FAERS_MIMIC_QID_COLS),
        ("vaers_mimic", "VAERS+MIMIC", vaers_mimic, ["VAERS", "MIMIC"], VAERS_MIMIC_QID_COLS),
    ]
    matched_tables: dict[str, pl.DataFrame] = {}
    with progress_bar(len(matched_specs), "Match records", "scope", progress, leave=True) as bar:
        for key, label, candidate_df, sources, qid_cols in matched_specs:
            bar.set_postfix_str(label)
            matched_tables[key] = matched_records_for_candidates(records, candidate_df, sources, qid_cols)
            bar.update(1)
    matched = matched_tables["matched"]
    faers_vaers_matched = matched_tables["faers_vaers"]
    faers_mimic_matched = matched_tables["faers_mimic"]
    vaers_mimic_matched = matched_tables["vaers_mimic"]

    t0 = now()
    probability_specs = [
        (
            "three",
            "FAERS+VAERS+MIMIC",
            matched if not matched.is_empty() else records,
            THREE_SOURCE_QID_COLS,
            "three_source_full_qid",
        ),
        (
            "faers_vaers",
            "FAERS+VAERS",
            faers_vaers_matched if not faers_vaers_matched.is_empty() else records.filter(pl.col("source").is_in(["FAERS", "VAERS"])),
            FAERS_VAERS_QID_COLS,
            "faers_vaers_full_qid",
        ),
        (
            "faers_mimic",
            "FAERS+MIMIC",
            faers_mimic_matched if not faers_mimic_matched.is_empty() else records.filter(pl.col("source").is_in(["FAERS", "MIMIC"])),
            FAERS_MIMIC_QID_COLS,
            "faers_mimic_full_qid",
        ),
        (
            "vaers_mimic",
            "VAERS+MIMIC",
            vaers_mimic_matched if not vaers_mimic_matched.is_empty() else records.filter(pl.col("source").is_in(["VAERS", "MIMIC"])),
            VAERS_MIMIC_QID_COLS,
            "vaers_mimic_full_qid",
        ),
    ]
    probability_tables: dict[str, pl.DataFrame] = {}
    with progress_bar(len(probability_specs), "Compute probabilities", "scope", progress, leave=True) as bar:
        for key, label, scoped_records, qid_cols, scope_name in probability_specs:
            bar.set_postfix_str(label)
            probability_tables[key] = probability_table(scoped_records, sensitive, qid_cols, scope_name)
            bar.update(1)
    probabilities_three = probability_tables["three"]
    probabilities_faers_vaers = probability_tables["faers_vaers"]
    probabilities_faers_mimic = probability_tables["faers_mimic"]
    probabilities_vaers_mimic = probability_tables["vaers_mimic"]
    probabilities = concat_frames([probabilities_three, probabilities_faers_vaers, probabilities_faers_mimic, probabilities_vaers_mimic])
    log(f"computed probabilities in {now() - t0:.1f}s")

    t0 = now()
    protected, protected_sensitive, suppressed_sensitive_three = apply_c_bounding(
        matched if not matched.is_empty() else records,
        sensitive,
        THREE_SOURCE_QID_COLS,
        c,
        max_iterations,
        progress,
        "C-bound FAERS+VAERS+MIMIC",
    )
    protected_faers_vaers, protected_sensitive_faers_vaers, suppressed_sensitive_faers_vaers = apply_c_bounding(
        faers_vaers_matched if not faers_vaers_matched.is_empty() else records.filter(pl.col("source").is_in(["FAERS", "VAERS"])),
        sensitive,
        FAERS_VAERS_QID_COLS,
        c,
        max_iterations,
        progress,
        "C-bound FAERS+VAERS",
    )
    protected_faers_mimic, protected_sensitive_faers_mimic, suppressed_sensitive_faers_mimic = apply_c_bounding(
        faers_mimic_matched if not faers_mimic_matched.is_empty() else records.filter(pl.col("source").is_in(["FAERS", "MIMIC"])),
        sensitive,
        FAERS_MIMIC_QID_COLS,
        c,
        max_iterations,
        progress,
        "C-bound FAERS+MIMIC",
    )
    protected_vaers_mimic, protected_sensitive_vaers_mimic, suppressed_sensitive_vaers_mimic = apply_c_bounding(
        vaers_mimic_matched if not vaers_mimic_matched.is_empty() else records.filter(pl.col("source").is_in(["VAERS", "MIMIC"])),
        sensitive,
        VAERS_MIMIC_QID_COLS,
        c,
        max_iterations,
        progress,
        "C-bound VAERS+MIMIC",
    )
    protected_prob_specs = [
        ("FAERS+VAERS+MIMIC", protected, protected_sensitive, THREE_SOURCE_QID_COLS, "three_source_full_qid_protected"),
        ("FAERS+VAERS", protected_faers_vaers, protected_sensitive_faers_vaers, FAERS_VAERS_QID_COLS, "faers_vaers_full_qid_protected"),
        ("FAERS+MIMIC", protected_faers_mimic, protected_sensitive_faers_mimic, FAERS_MIMIC_QID_COLS, "faers_mimic_full_qid_protected"),
        ("VAERS+MIMIC", protected_vaers_mimic, protected_sensitive_vaers_mimic, VAERS_MIMIC_QID_COLS, "vaers_mimic_full_qid_protected"),
    ]
    protected_prob_frames: list[pl.DataFrame] = []
    with progress_bar(len(protected_prob_specs), "Protected probabilities", "scope", progress, leave=True) as bar:
        for label, protected_records, protected_sensitive_rows, qid_cols, scope_name in protected_prob_specs:
            bar.set_postfix_str(label)
            protected_prob_frames.append(probability_table(protected_records, protected_sensitive_rows, qid_cols, scope_name))
            bar.update(1)
    protected_prob = concat_frames(protected_prob_frames)
    log(f"applied C-bounding in {now() - t0:.1f}s")

    t0 = now()
    output_specs = [
        ("source_records.csv", records),
        ("qid_field_inventory.csv", qid_inventory),
        ("intersection_three_source_candidates.csv", three),
        ("intersection_faers_vaers_candidates.csv", faers_vaers),
        ("intersection_faers_mimic_candidates.csv", faers_mimic),
        ("intersection_vaers_mimic_candidates.csv", vaers_mimic),
        ("matched_patient_details.csv", matched),
        ("qid_sensitive_probability.csv", probabilities),
        ("protected_c_bounded_records.csv", protected),
        ("protected_faers_vaers_c_bounded_records.csv", protected_faers_vaers),
        ("protected_faers_mimic_c_bounded_records.csv", protected_faers_mimic),
        ("protected_vaers_mimic_c_bounded_records.csv", protected_vaers_mimic),
        ("protected_c_bounded_probability.csv", protected_prob),
    ]
    with progress_bar(len(output_specs), "Write CSV", "file", progress, leave=True) as bar:
        for filename, df in output_specs:
            bar.set_postfix_str(filename)
            write_csv(df, output_dir / filename)
            bar.update(1)
    log(f"wrote CSV outputs in {now() - t0:.1f}s")

    source_counts = dict(records.group_by("source").len("count").iter_rows())
    matched_counts = dict(matched.group_by("source").len("count").iter_rows()) if not matched.is_empty() else {}
    protected_counts = dict(protected.group_by("source").len("count").iter_rows()) if not protected.is_empty() else {}
    protected_three_prob = protected_prob.filter(pl.col("qid_scope") == "three_source_full_qid_protected")

    summary = {
        "engine": {
            "dataframe": "polars",
            "arrow": pa.__version__,
            "csv_backend": backend,
            "cudf_available": has_cudf(),
        },
        "c_bound": c,
        "qid_fields": {
            "three_source": THREE_SOURCE_QID_COLS,
            "faers_vaers": FAERS_VAERS_QID_COLS,
            "faers_mimic": FAERS_MIMIC_QID_COLS,
            "vaers_mimic": VAERS_MIMIC_QID_COLS,
        },
        "records_by_source": {str(k): int(v) for k, v in source_counts.items()},
        "total_source_records": int(records.height),
        "sensitive_rows": int(sensitive.height),
        "three_source_qid_groups": int(three.height),
        "faers_vaers_full_qid_groups": int(faers_vaers.height),
        "faers_mimic_full_qid_groups": int(faers_mimic.height),
        "vaers_mimic_full_qid_groups": int(vaers_mimic.height),
        "matched_detail_records": int(matched.height),
        "matched_detail_records_by_source": {str(k): int(v) for k, v in matched_counts.items()},
        "three_source_candidate_combinations": int(three.select(pl.col("candidate_combination_count").sum()).item() or 0) if "candidate_combination_count" in three.columns else 0,
        "faers_vaers_candidate_pairs": int(faers_vaers.select(pl.col("candidate_pair_count").sum()).item() or 0) if "candidate_pair_count" in faers_vaers.columns else 0,
        "faers_mimic_candidate_pairs": int(faers_mimic.select(pl.col("candidate_pair_count").sum()).item() or 0) if "candidate_pair_count" in faers_mimic.columns else 0,
        "vaers_mimic_candidate_pairs": int(vaers_mimic.select(pl.col("candidate_pair_count").sum()).item() or 0) if "candidate_pair_count" in vaers_mimic.columns else 0,
        "probability_rows": int(probabilities.height),
        "max_probability_before_protection_all_sources": max_probability(probabilities_three, "ALL"),
        "protected_records": int(protected.height),
        "protected_records_by_source": {str(k): int(v) for k, v in protected_counts.items()},
        "suppressed_records_for_c_bounding": int((matched.height if not matched.is_empty() else records.height) - protected.height),
        "suppressed_sensitive_values_by_scope": {
            "three_source": int(suppressed_sensitive_three),
            "faers_vaers": int(suppressed_sensitive_faers_vaers),
            "faers_mimic": int(suppressed_sensitive_faers_mimic),
            "vaers_mimic": int(suppressed_sensitive_vaers_mimic),
        },
        "protected_records_by_scope": {
            "three_source": int(protected.height),
            "faers_vaers": int(protected_faers_vaers.height),
            "faers_mimic": int(protected_faers_mimic.height),
            "vaers_mimic": int(protected_vaers_mimic.height),
        },
        "suppressed_records_by_scope": {
            "three_source": int((matched.height if not matched.is_empty() else records.height) - protected.height),
            "faers_vaers": int((faers_vaers_matched.height if not faers_vaers_matched.is_empty() else records.filter(pl.col("source").is_in(["FAERS", "VAERS"])).height) - protected_faers_vaers.height),
            "faers_mimic": int((faers_mimic_matched.height if not faers_mimic_matched.is_empty() else records.filter(pl.col("source").is_in(["FAERS", "MIMIC"])).height) - protected_faers_mimic.height),
            "vaers_mimic": int((vaers_mimic_matched.height if not vaers_mimic_matched.is_empty() else records.filter(pl.col("source").is_in(["VAERS", "MIMIC"])).height) - protected_vaers_mimic.height),
        },
        "max_probability_after_protection_all_sources": max_probability(protected_three_prob, "ALL"),
        "max_probability_after_protection_all_qid_scopes": max_probability(protected_prob, "ALL"),
        "outputs": {
            "source_records": str(output_dir / "source_records.csv"),
            "qid_field_inventory": str(output_dir / "qid_field_inventory.csv"),
            "three_source_candidates": str(output_dir / "intersection_three_source_candidates.csv"),
            "faers_vaers_candidates": str(output_dir / "intersection_faers_vaers_candidates.csv"),
            "faers_mimic_candidates": str(output_dir / "intersection_faers_mimic_candidates.csv"),
            "vaers_mimic_candidates": str(output_dir / "intersection_vaers_mimic_candidates.csv"),
            "matched_patient_details": str(output_dir / "matched_patient_details.csv"),
            "qid_sensitive_probability": str(output_dir / "qid_sensitive_probability.csv"),
            "protected_c_bounded_records": str(output_dir / "protected_c_bounded_records.csv"),
            "protected_faers_vaers_c_bounded_records": str(output_dir / "protected_faers_vaers_c_bounded_records.csv"),
            "protected_faers_mimic_c_bounded_records": str(output_dir / "protected_faers_mimic_c_bounded_records.csv"),
            "protected_vaers_mimic_c_bounded_records": str(output_dir / "protected_vaers_mimic_c_bounded_records.csv"),
            "protected_c_bounded_probability": str(output_dir / "protected_c_bounded_probability.csv"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    total_started = now()
    data_root = Path(args.data_root)
    progress = not args.no_progress
    records, sensitive = normalize_records(data_root, args.backend, args.cudf_min_mb, progress)
    summary = write_outputs(
        records,
        sensitive,
        Path(args.output_dir),
        args.c,
        args.top_sensitive,
        args.max_c_bound_iterations,
        args.backend,
        progress,
    )
    summary["elapsed_seconds"] = round(now() - total_started, 3)
    (Path(args.output_dir) / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
