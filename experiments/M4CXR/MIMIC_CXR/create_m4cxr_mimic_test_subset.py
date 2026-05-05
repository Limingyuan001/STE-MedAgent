#!/usr/bin/env python3
"""Reconstruct the M4CXR MIMIC-CXR single-image test subset and optionally download it.

This script follows the M4CXR preprocessing logic at the dicom level:
- use official split metadata to keep only the `test` split
- use sectioned reports to attach study-level `findings`
- expand to dicom-level rows from the official metadata CSV
- keep only rows whose study has non-empty findings

Outputs are written under `/data/lmy/datasets/M4CXR_MIMIC` by default:
- manifests/m4cxr_mimic_test_3858.jsonl
- manifests/m4cxr_mimic_test_2461_frontal.jsonl
- manifests/m4cxr_mimic_test_3858_jpg_urls.txt
- manifests/m4cxr_mimic_test_3858_report_urls.txt
- reports_raw/
- jpg_raw/
- logs/
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import shutil
import subprocess
import urllib.request
from pathlib import Path
from typing import Any, Iterable, Iterator, TextIO

try:
    from section_parser import section_text
except ImportError:
    from experiments.M4CXR.MIMIC_CXR.section_parser import section_text  # type: ignore


DEFAULT_OUTPUT_ROOT = Path("/data/lmy/datasets/M4CXR_MIMIC")
DEFAULT_REPORTS_ROOT = Path("/data/lmy/datasets/M4CXR_MIMIC/mimic-cxr-reports")
DEFAULT_REPORT_BASE = "https://physionet.org/files/mimic-cxr/2.1.0"
DEFAULT_JPG_BASE = "https://physionet.org/files/mimic-cxr-jpg/2.1.0"

FULL_EXPECTED_COUNT = 3858
FRONTAL_EXPECTED_COUNT = 2461
NON_FRONTAL_EXPECTED_COUNT = 1397
FRONTAL_VIEWS = {"PA", "AP"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct the M4CXR MIMIC-CXR single-image test subset from official "
            "metadata CSVs and optionally download the source JPGs and reports."
        ),
    )
    parser.add_argument(
        "--sectioned-csv",
        type=Path,
        help="Path to sectioned report CSV (for example mimic_cxr_sectioned.csv).",
    )
    parser.add_argument(
        "--reports-root",
        type=Path,
        default=DEFAULT_REPORTS_ROOT,
        help=(
            "Root directory of extracted MIMIC-CXR text reports. Used when "
            "--sectioned-csv is not provided. Default: "
            f"{DEFAULT_REPORTS_ROOT}"
        ),
    )
    parser.add_argument(
        "--split-csv",
        type=Path,
        required=True,
        help="Path to official split CSV (for example mimic-cxr-2.0.0-split.csv.gz).",
    )
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        required=True,
        help="Path to official metadata CSV (for example mimic-cxr-2.0.0-metadata.csv.gz).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Output root directory. Default: {DEFAULT_OUTPUT_ROOT}",
    )
    parser.add_argument(
        "--download",
        choices=("none", "reports", "jpg", "all"),
        default="none",
        help="Download mode. Default: none",
    )
    parser.add_argument(
        "--physionet-base-report",
        default=DEFAULT_REPORT_BASE,
        help=f"Base URL for MIMIC-CXR reports. Default: {DEFAULT_REPORT_BASE}",
    )
    parser.add_argument(
        "--physionet-base-jpg",
        default=DEFAULT_JPG_BASE,
        help=f"Base URL for MIMIC-CXR JPG images. Default: {DEFAULT_JPG_BASE}",
    )
    return parser.parse_args()


def open_text_maybe_gzip(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def normalize_study_id(value: Any) -> str:
    study_id = str(value).strip()
    if study_id.startswith("s"):
        study_id = study_id[1:]
    return study_id


def normalize_string(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def normalize_findings(value: Any) -> str:
    findings = normalize_string(value)
    if findings.lower() == "nan":
        return ""
    return findings


def subject_prefix(subject_id: str) -> str:
    if len(subject_id) < 2:
        raise ValueError(f"Invalid subject_id '{subject_id}'")
    return subject_id[:2]


def join_url(base: str, relpath: str) -> str:
    return base.rstrip("/") + "/" + relpath.lstrip("/")


def load_split_map(split_csv_path: Path) -> dict[str, str]:
    split_map: dict[str, str] = {}
    with open_text_maybe_gzip(split_csv_path) as handle:
        reader = csv.DictReader(handle)
        required = {"study_id", "split"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(
                f"{split_csv_path} missing required columns {sorted(required)}; "
                f"got {reader.fieldnames}"
            )

        for row in reader:
            study_id = normalize_study_id(row["study_id"])
            split_value = normalize_string(row["split"])
            if not study_id:
                continue
            if study_id in split_map and split_map[study_id] != split_value:
                raise ValueError(
                    f"Inconsistent split for study_id {study_id}: "
                    f"{split_map[study_id]} vs {split_value}"
                )
            split_map[study_id] = split_value

    if not split_map:
        raise ValueError(f"No split rows loaded from {split_csv_path}")
    return split_map


def load_sectioned_findings(sectioned_csv_path: Path) -> dict[str, str]:
    findings_map: dict[str, str] = {}
    with open_text_maybe_gzip(sectioned_csv_path) as handle:
        reader = csv.DictReader(handle)
        required = {"study", "findings"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(
                f"{sectioned_csv_path} missing required columns {sorted(required)}; "
                f"got {reader.fieldnames}"
            )

        for row in reader:
            study_id = normalize_study_id(row["study"])
            findings = normalize_findings(row["findings"])
            if not study_id:
                continue
            findings_map[study_id] = findings

    if not findings_map:
        raise ValueError(f"No sectioned report rows loaded from {sectioned_csv_path}")
    return findings_map


def list_rindex(items: list[str], value: str) -> int:
    return len(items) - items[-1::-1].index(value) - 1


def extract_report_findings(report_text: str, study_stem: str) -> str:
    del study_stem
    sections, section_names, _section_idx = section_text(report_text)
    if "findings" not in section_names:
        return ""
    idx = list_rindex(section_names, "findings")
    return normalize_findings(sections[idx])


def load_findings_from_reports(reports_root: Path) -> dict[str, str]:
    if not reports_root.exists():
        raise FileNotFoundError(f"Reports root not found: {reports_root}")

    findings_map: dict[str, str] = {}
    group_dirs = sorted(
        path for path in reports_root.iterdir() if path.is_dir() and path.name.startswith("p")
    )
    if not group_dirs:
        raise ValueError(f"No report group directories found under {reports_root}")

    for group_dir in group_dirs:
        patient_dirs = sorted(
            path for path in group_dir.iterdir() if path.is_dir() and path.name.startswith("p")
        )
        for patient_dir in patient_dirs:
            report_files = sorted(patient_dir.glob("s*.txt"))
            for report_file in report_files:
                report_text = report_file.read_text(encoding="utf-8")
                study_stem = report_file.stem
                findings = extract_report_findings(report_text, study_stem)
                findings_map[normalize_study_id(study_stem)] = normalize_findings(findings)

    if not findings_map:
        raise ValueError(f"No reports were parsed from {reports_root}")
    return findings_map


def iter_metadata_rows(metadata_csv_path: Path) -> Iterator[dict[str, str]]:
    with open_text_maybe_gzip(metadata_csv_path) as handle:
        reader = csv.DictReader(handle)
        required = {"subject_id", "study_id", "dicom_id", "ViewPosition"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(
                f"{metadata_csv_path} missing required columns {sorted(required)}; "
                f"got {reader.fieldnames}"
            )
        for row in reader:
            yield row


def build_manifest(
    split_map: dict[str, str],
    findings_map: dict[str, str],
    metadata_rows: Iterable[dict[str, str]],
    report_base: str,
    jpg_base: str,
) -> list[dict[str, str]]:
    manifest: list[dict[str, str]] = []

    for row in metadata_rows:
        subject_id = normalize_string(row["subject_id"])
        study_id = normalize_study_id(row["study_id"])
        dicom_stem = normalize_string(row["dicom_id"])
        view_position = normalize_string(row["ViewPosition"])

        if not subject_id or not study_id or not dicom_stem:
            continue

        split_value = split_map.get(study_id)
        if split_value != "test":
            continue

        findings = findings_map.get(study_id, "")
        if not findings:
            continue

        dicom_id = dicom_stem if dicom_stem.endswith(".jpg") else f"{dicom_stem}.jpg"
        prefix = subject_prefix(subject_id)
        report_relpath = f"files/p{prefix}/p{subject_id}/s{study_id}.txt"
        jpg_relpath = f"files/p{prefix}/p{subject_id}/s{study_id}/{dicom_id}"

        manifest.append(
            {
                "subject_id": subject_id,
                "study_id": study_id,
                "dicom_id": dicom_id,
                "view_position": view_position,
                "findings": findings,
                "report_relpath": report_relpath,
                "jpg_relpath": jpg_relpath,
                "report_url": join_url(report_base, report_relpath),
                "jpg_url": join_url(jpg_base, jpg_relpath),
            }
        )

    if not manifest:
        raise ValueError("No dicom-level test rows were generated. Check the input CSVs.")
    return manifest


def filter_frontal(manifest: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in manifest if row["view_position"] in FRONTAL_VIEWS]


def validate_counts(full_manifest: list[dict[str, str]], frontal_manifest: list[dict[str, str]]) -> None:
    full_count = len(full_manifest)
    frontal_count = len(frontal_manifest)
    non_frontal_count = full_count - frontal_count

    if full_count != FULL_EXPECTED_COUNT:
        raise ValueError(
            f"Full manifest count mismatch: got {full_count}, expected {FULL_EXPECTED_COUNT}."
        )
    if frontal_count != FRONTAL_EXPECTED_COUNT:
        raise ValueError(
            f"Frontal manifest count mismatch: got {frontal_count}, expected {FRONTAL_EXPECTED_COUNT}."
        )
    if non_frontal_count != NON_FRONTAL_EXPECTED_COUNT:
        raise ValueError(
            "Non-frontal manifest count mismatch: "
            f"got {non_frontal_count}, expected {NON_FRONTAL_EXPECTED_COUNT}."
        )


def ensure_directories(output_root: Path) -> dict[str, Path]:
    paths = {
        "output_root": output_root,
        "manifest_dir": output_root / "manifests",
        "reports_dir": output_root / "reports_raw",
        "jpg_dir": output_root / "jpg_raw",
        "logs_dir": output_root / "logs",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_url_list(path: Path, urls: Iterable[str]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for url in urls:
            handle.write(url)
            handle.write("\n")


def unique_by_relpath(
    rows: Iterable[dict[str, str]],
    relpath_key: str,
    url_key: str,
) -> list[dict[str, str]]:
    unique_rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in rows:
        relpath = row[relpath_key]
        if relpath in seen:
            continue
        seen.add(relpath)
        unique_rows.append({"relpath": relpath, "url": row[url_key]})
    return unique_rows


def download_with_curl(url: str, destination: Path) -> None:
    subprocess.run(
        [
            "curl",
            "-L",
            "--fail",
            "--silent",
            "--show-error",
            "-n",
            "-o",
            str(destination),
            url,
        ],
        check=True,
    )


def download_with_urllib(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request) as response, destination.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def download_file(url: str, destination: Path) -> None:
    if destination.exists():
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_suffix(destination.suffix + ".part")
    if tmp_path.exists():
        tmp_path.unlink()

    try:
        if shutil.which("curl"):
            download_with_curl(url, tmp_path)
        else:
            download_with_urllib(url, tmp_path)
        tmp_path.replace(destination)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def write_failure_log(path: Path, failures: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for item in failures:
            handle.write(f"{item['relpath']}\t{item['url']}\t{item['error']}\n")


def maybe_download(
    mode: str,
    full_manifest: list[dict[str, str]],
    reports_dir: Path,
    jpg_dir: Path,
    logs_dir: Path,
) -> None:
    download_reports = mode in {"reports", "all"}
    download_jpg = mode in {"jpg", "all"}

    if download_reports:
        report_entries = unique_by_relpath(full_manifest, "report_relpath", "report_url")
        failures: list[dict[str, str]] = []
        for entry in report_entries:
            destination = reports_dir / entry["relpath"]
            try:
                download_file(entry["url"], destination)
            except Exception as exc:
                failures.append(
                    {
                        "relpath": entry["relpath"],
                        "url": entry["url"],
                        "error": str(exc),
                    }
                )
        write_failure_log(logs_dir / "failed_reports.txt", failures)

    if download_jpg:
        jpg_entries = unique_by_relpath(full_manifest, "jpg_relpath", "jpg_url")
        failures = []
        for entry in jpg_entries:
            destination = jpg_dir / entry["relpath"]
            try:
                download_file(entry["url"], destination)
            except Exception as exc:
                failures.append(
                    {
                        "relpath": entry["relpath"],
                        "url": entry["url"],
                        "error": str(exc),
                    }
                )
        write_failure_log(logs_dir / "failed_jpg.txt", failures)


def main() -> None:
    args = parse_args()
    paths = ensure_directories(args.output_root)

    if args.sectioned_csv is None and args.reports_root is None:
        raise ValueError("Provide either --sectioned-csv or --reports-root.")

    split_map = load_split_map(args.split_csv)
    if args.sectioned_csv is not None:
        findings_map = load_sectioned_findings(args.sectioned_csv)
        findings_source = f"sectioned CSV: {args.sectioned_csv}"
    else:
        findings_map = load_findings_from_reports(args.reports_root)
        findings_source = f"reports root: {args.reports_root}"
    full_manifest = build_manifest(
        split_map=split_map,
        findings_map=findings_map,
        metadata_rows=iter_metadata_rows(args.metadata_csv),
        report_base=args.physionet_base_report,
        jpg_base=args.physionet_base_jpg,
    )
    frontal_manifest = filter_frontal(full_manifest)
    validate_counts(full_manifest, frontal_manifest)

    full_manifest_path = paths["manifest_dir"] / "m4cxr_mimic_test_3858.jsonl"
    frontal_manifest_path = paths["manifest_dir"] / "m4cxr_mimic_test_2461_frontal.jsonl"
    jpg_urls_path = paths["manifest_dir"] / "m4cxr_mimic_test_3858_jpg_urls.txt"
    jpg_relpaths_path = paths["manifest_dir"] / "m4cxr_mimic_test_3858_jpg_relpaths.txt"
    report_urls_path = paths["manifest_dir"] / "m4cxr_mimic_test_3858_report_urls.txt"
    report_relpaths_path = paths["manifest_dir"] / "m4cxr_mimic_test_3858_report_relpaths.txt"

    write_jsonl(full_manifest_path, full_manifest)
    write_jsonl(frontal_manifest_path, frontal_manifest)
    write_url_list(jpg_urls_path, (row["jpg_url"] for row in full_manifest))
    write_url_list(jpg_relpaths_path, (row["jpg_relpath"] for row in full_manifest))
    write_url_list(
        report_urls_path,
        (row["url"] for row in unique_by_relpath(full_manifest, "report_relpath", "report_url")),
    )
    write_url_list(
        report_relpaths_path,
        (row["relpath"] for row in unique_by_relpath(full_manifest, "report_relpath", "report_url")),
    )

    maybe_download(
        mode=args.download,
        full_manifest=full_manifest,
        reports_dir=paths["reports_dir"],
        jpg_dir=paths["jpg_dir"],
        logs_dir=paths["logs_dir"],
    )

    unique_reports = len(unique_by_relpath(full_manifest, "report_relpath", "report_url"))
    print(f"Findings source: {findings_source}")
    print(f"Split CSV: {args.split_csv}")
    print(f"Metadata CSV: {args.metadata_csv}")
    print(f"Output root: {args.output_root}")
    print(f"Full manifest: {full_manifest_path} ({len(full_manifest)} rows)")
    print(f"Frontal manifest: {frontal_manifest_path} ({len(frontal_manifest)} rows)")
    print(
        "View counts: "
        f"frontal={len(frontal_manifest)}, non_frontal={len(full_manifest) - len(frontal_manifest)}"
    )
    print(f"Unique reports: {unique_reports}")
    print(f"Download mode: {args.download}")


if __name__ == "__main__":
    main()
