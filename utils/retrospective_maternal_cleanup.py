from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable


START_DATE = date(2025, 5, 29)
END_DATE = date(2025, 8, 26)


def _normalize_col_name(value: str) -> str:
    return value.strip().lower().replace(" ", "_")


def _find_col_index(header: list[str], candidates: Iterable[str]) -> int:
    normalized_to_original = {_normalize_col_name(c): c for c in candidates}
    for idx, col in enumerate(header):
        if _normalize_col_name(col) in normalized_to_original:
            return idx
    raise KeyError(
        f"None of columns {list(candidates)!r} found in header: {header!r}"
    )


def _parse_date(value: str) -> date | None:
    value = (value or "").strip()
    if not value:
        return None

    # Fast path for ISO-like timestamps: "YYYY-MM-DD..." (e.g. 2025-05-29T00:00:00.000)
    if len(value) >= 10 and value[4] == "-" and value[7] == "-":
        try:
            return datetime.strptime(value[:10], "%Y-%m-%d").date()
        except ValueError:
            pass

    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    return None


def _default_maternal_dir() -> Path:
    here = Path(__file__).resolve()
    candidates = [
        here.parent / "imnid" / "maternal",  # if script is in utils/
        here.parent / "utils" / "imnid" / "maternal",  # if script is in repo root
        Path.cwd() / "utils" / "imnid" / "maternal",
        Path.cwd() / "imnid" / "maternal",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        "Could not locate the maternal IMNID folder. Tried: "
        + ", ".join(str(c) for c in candidates)
    )


def _read_header(path: Path) -> list[str]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        try:
            return next(reader)
        except StopIteration as exc:
            raise ValueError(f"{path} is empty") from exc


def _union_fieldnames(header_a: list[str], header_b: list[str]) -> list[str]:
    # Use the national file as the baseline ordering, then append any extra columns.
    seen = set()
    out: list[str] = []
    for col in header_a:
        if col not in seen:
            out.append(col)
            seen.add(col)
    for col in header_b:
        if col not in seen:
            out.append(col)
            seen.add(col)
    return out


@dataclass(frozen=True)
class ProcessResult:
    kept: int
    removed: int


def _process_csv(
    *,
    input_path: Path,
    output_tmp_path: Path,
    fieldnames: list[str],
    retrospective_writer: csv.DictWriter,
    union_fieldnames: list[str],
    source_file: str,
    start_date: date,
    end_date: date,
    require_region_oromia: bool,
    dry_run: bool,
) -> ProcessResult:
    enrollment_idx = _find_col_index(fieldnames, ["enrollment_date", "enrollment date"])
    region_idx = None
    if require_region_oromia:
        region_idx = _find_col_index(fieldnames, ["region_name", "region name"])

    kept = 0
    removed = 0

    fin = input_path.open("r", newline="", encoding="utf-8-sig")
    try:
        reader = csv.reader(fin)
        header = next(reader)
        if header != fieldnames:
            # Defensive: if the file header changed between the header-read step and now.
            raise ValueError(
                f"Header mismatch for {input_path}. Expected {fieldnames!r} but got {header!r}"
            )

        if dry_run:
            for row in reader:
                if len(row) < len(fieldnames):
                    row = row + [""] * (len(fieldnames) - len(row))
                remove_row = False
                enrollment = _parse_date(row[enrollment_idx])
                if enrollment is not None and start_date <= enrollment <= end_date:
                    if require_region_oromia:
                        region_val = (row[region_idx] if region_idx is not None else "").strip()
                        remove_row = region_val.casefold() == "oromia"
                    else:
                        remove_row = True

                if remove_row:
                    removed += 1
                else:
                    kept += 1
            return ProcessResult(kept=kept, removed=removed)

        fout = output_tmp_path.open("w", newline="", encoding="utf-8")
        try:
            writer = csv.writer(fout)
            writer.writerow(fieldnames)

            for row in reader:
                if len(row) < len(fieldnames):
                    row = row + [""] * (len(fieldnames) - len(row))

                remove_row = False
                enrollment = _parse_date(row[enrollment_idx])
                if enrollment is not None and start_date <= enrollment <= end_date:
                    if require_region_oromia:
                        region_val = (row[region_idx] if region_idx is not None else "").strip()
                        remove_row = region_val.casefold() == "oromia"
                    else:
                        remove_row = True

                if remove_row:
                    removed += 1
                    row_dict = {k: v for k, v in zip(fieldnames, row)}
                    out_row = {k: row_dict.get(k, "") for k in union_fieldnames}
                    out_row["source_file"] = source_file
                    retrospective_writer.writerow(out_row)
                else:
                    kept += 1
                    writer.writerow(row)
        finally:
            fout.close()
    finally:
        fin.close()

    return ProcessResult(kept=kept, removed=removed)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Remove retrospective rows from maternal IMNID CSVs and archive them "
            "into retrospective_data.csv."
        )
    )
    parser.add_argument(
        "--maternal-dir",
        type=Path,
        default=None,
        help="Path to the maternal IMNID folder (default: auto-detect).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute counts only; do not write/modify any files.",
    )
    args = parser.parse_args()

    maternal_dir = args.maternal_dir or _default_maternal_dir()
    national_path = maternal_dir / "national_maternal.csv"
    regional_path = maternal_dir / "regional_Oromia_maternal.csv"
    retrospective_path = maternal_dir / "retrospective_data.csv"

    if not national_path.is_file():
        raise FileNotFoundError(f"Missing file: {national_path}")
    if not regional_path.is_file():
        raise FileNotFoundError(f"Missing file: {regional_path}")

    national_header = _read_header(national_path)
    regional_header = _read_header(regional_path)
    union_cols = _union_fieldnames(national_header, regional_header)
    retrospective_header = ["source_file", *union_cols]

    if args.dry_run:
        # We still use the processing function for parity, but don't open the retrospective file.
        dummy_sink = Path(os.devnull)
        with dummy_sink.open("w", newline="", encoding="utf-8") as devnull:
            retrospective_writer = csv.DictWriter(
                devnull, fieldnames=retrospective_header, extrasaction="ignore"
            )
            national_res = _process_csv(
                input_path=national_path,
                output_tmp_path=national_path.with_suffix(".tmp"),
                fieldnames=national_header,
                retrospective_writer=retrospective_writer,
                union_fieldnames=retrospective_header,
                source_file=national_path.name,
                start_date=START_DATE,
                end_date=END_DATE,
                require_region_oromia=True,
                dry_run=True,
            )
            regional_res = _process_csv(
                input_path=regional_path,
                output_tmp_path=regional_path.with_suffix(".tmp"),
                fieldnames=regional_header,
                retrospective_writer=retrospective_writer,
                union_fieldnames=retrospective_header,
                source_file=regional_path.name,
                start_date=START_DATE,
                end_date=END_DATE,
                require_region_oromia=False,
                dry_run=True,
            )

        total_removed = national_res.removed + regional_res.removed
        print(f"[DRY RUN] maternal_dir: {maternal_dir}")
        print(
            f"[DRY RUN] national removed: {national_res.removed}, kept: {national_res.kept}"
        )
        print(
            f"[DRY RUN] regional Oromia removed: {regional_res.removed}, kept: {regional_res.kept}"
        )
        print(
            f"[DRY RUN] would write {total_removed} removed rows to: {retrospective_path}"
        )
        return 0

    national_tmp = national_path.with_suffix(".csv.tmp")
    regional_tmp = regional_path.with_suffix(".csv.tmp")
    retrospective_tmp = retrospective_path.with_suffix(".csv.tmp")

    with retrospective_tmp.open("w", newline="", encoding="utf-8") as fout:
        retrospective_writer = csv.DictWriter(
            fout, fieldnames=retrospective_header, extrasaction="ignore"
        )
        retrospective_writer.writeheader()

        national_res = _process_csv(
            input_path=national_path,
            output_tmp_path=national_tmp,
            fieldnames=national_header,
            retrospective_writer=retrospective_writer,
            union_fieldnames=retrospective_header,
            source_file=national_path.name,
            start_date=START_DATE,
            end_date=END_DATE,
            require_region_oromia=True,
            dry_run=False,
        )
        regional_res = _process_csv(
            input_path=regional_path,
            output_tmp_path=regional_tmp,
            fieldnames=regional_header,
            retrospective_writer=retrospective_writer,
            union_fieldnames=retrospective_header,
            source_file=regional_path.name,
            start_date=START_DATE,
            end_date=END_DATE,
            require_region_oromia=False,
            dry_run=False,
        )

    os.replace(national_tmp, national_path)
    os.replace(regional_tmp, regional_path)
    os.replace(retrospective_tmp, retrospective_path)

    total_removed = national_res.removed + regional_res.removed
    print(f"maternal_dir: {maternal_dir}")
    print(f"removed rows written to: {retrospective_path} ({total_removed} rows)")
    print(f"national removed: {national_res.removed}, kept: {national_res.kept}")
    print(
        f"regional Oromia removed: {regional_res.removed}, kept: {regional_res.kept}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

