from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import bcrypt
import pandas as pd
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKBOOK = REPO_ROOT / "utils" / "DQO_users.xlsx"
DQ_OFFICER_ROLE = "dq_officer"
REQUIRED_COLUMNS = {"DQO_name", "user name", "PASSWORD", "region", "facility"}

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

load_dotenv(REPO_ROOT / ".env")

from utils.db import get_db_connection


@dataclass
class DQOUserRecord:
    username: str
    password: str
    full_name: str
    first_name: str
    last_name: str
    region_name: str
    facility_names: list[str]


def normalize_text(value: object) -> str:
    text = "" if value is None else str(value)
    return re.sub(r"\s+", " ", text).strip().lower()


def split_name(full_name: str) -> tuple[str, str]:
    parts = [part for part in str(full_name).strip().split() if part]
    if not parts:
        return "", ""
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], " ".join(parts[1:])


def unique_preserving_order(values: Iterable[object]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        text = str(value).strip()
        if not text:
            continue
        key = normalize_text(text)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(text)
    return ordered


def load_records(workbook_path: Path, username_filter: str | None = None) -> list[DQOUserRecord]:
    df = pd.read_excel(workbook_path)
    df.columns = [str(column).strip() for column in df.columns]

    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"Workbook is missing required columns: {missing_list}")

    df = df.fillna("")
    df["user name"] = df["user name"].astype(str).str.strip()
    df = df[df["user name"] != ""].copy()

    if username_filter:
        df = df[
            df["user name"].str.strip().str.lower() == username_filter.strip().lower()
        ].copy()

    if df.empty:
        raise ValueError("No DQO user rows found in the workbook for the selected filter.")

    records: list[DQOUserRecord] = []
    for username, group in df.groupby("user name", sort=False):
        full_names = unique_preserving_order(group["DQO_name"].tolist())
        passwords = unique_preserving_order(group["PASSWORD"].tolist())
        regions = unique_preserving_order(group["region"].tolist())
        facilities = unique_preserving_order(group["facility"].tolist())

        if len(full_names) != 1:
            raise ValueError(
                f"Username '{username}' has inconsistent DQO_name values: {full_names}"
            )
        if len(passwords) != 1:
            raise ValueError(
                f"Username '{username}' has inconsistent PASSWORD values: {passwords}"
            )
        if len(regions) != 1:
            raise ValueError(
                f"Username '{username}' has inconsistent region values: {regions}"
            )
        if not facilities:
            raise ValueError(f"Username '{username}' has no facility rows.")

        first_name, last_name = split_name(full_names[0])
        records.append(
            DQOUserRecord(
                username=username.strip(),
                password=passwords[0],
                full_name=full_names[0],
                first_name=first_name,
                last_name=last_name,
                region_name=regions[0],
                facility_names=facilities,
            )
        )

    return records


def ensure_schema(cur) -> None:
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS user_facility_access (
            user_id INTEGER NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
            facility_id INTEGER NOT NULL REFERENCES facilities(facility_id) ON DELETE CASCADE,
            PRIMARY KEY (user_id, facility_id)
        )
        """
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_user_facility_access_user_id ON user_facility_access(user_id)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_user_facility_access_facility_id ON user_facility_access(facility_id)"
    )

    cur.execute("ALTER TABLE users DROP CONSTRAINT IF EXISTS facility_user_ck")
    cur.execute("ALTER TABLE users DROP CONSTRAINT IF EXISTS users_scope_check")
    cur.execute("ALTER TABLE users DROP CONSTRAINT IF EXISTS users_role_check")

    cur.execute(
        """
        ALTER TABLE users
        ADD CONSTRAINT users_role_check
        CHECK (role IN ('facility', 'regional', 'national', 'admin', 'dq_officer'))
        """
    )
    cur.execute(
        """
        ALTER TABLE users
        ADD CONSTRAINT users_scope_check CHECK (
            (role = 'facility'   AND facility_id IS NOT NULL AND region_id IS NULL     AND country_id IS NULL) OR
            (role = 'dq_officer' AND region_id   IS NOT NULL AND facility_id IS NULL   AND country_id IS NULL) OR
            (role = 'regional'   AND region_id   IS NOT NULL AND facility_id IS NULL   AND country_id IS NULL) OR
            (role = 'national'   AND country_id  IS NOT NULL AND facility_id IS NULL   AND region_id IS NULL) OR
            (role = 'admin'      AND facility_id IS NULL     AND region_id IS NULL     AND country_id IS NULL)
        )
        """
    )


def resolve_region(cur, region_name: str) -> tuple[int, str]:
    cur.execute("SELECT region_id, region_name FROM regions")
    rows = cur.fetchall()
    matches = [row for row in rows if normalize_text(row[1]) == normalize_text(region_name)]

    if not matches:
        raise ValueError(f"Region '{region_name}' was not found in the regions table.")
    if len(matches) > 1:
        raise ValueError(f"Region '{region_name}' matched multiple rows in regions.")

    return matches[0]


def resolve_facilities(
    cur, region_id: int, facility_names: list[str]
) -> list[tuple[int, str]]:
    cur.execute(
        """
        SELECT facility_id, facility_name
        FROM facilities
        WHERE region_id = %s
        ORDER BY facility_name
        """,
        (region_id,),
    )
    rows = cur.fetchall()

    lookup: dict[str, tuple[int, str]] = {}
    duplicates: set[str] = set()
    for facility_id, facility_name in rows:
        key = normalize_text(facility_name)
        if key in lookup:
            duplicates.add(key)
            continue
        lookup[key] = (facility_id, facility_name)

    if duplicates:
        dup_names = ", ".join(sorted(duplicates))
        raise ValueError(
            f"Duplicate normalized facility names found in region_id={region_id}: {dup_names}"
        )

    resolved: list[tuple[int, str]] = []
    missing: list[str] = []
    for facility_name in facility_names:
        match = lookup.get(normalize_text(facility_name))
        if not match:
            missing.append(facility_name)
            continue
        resolved.append(match)

    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(
            f"These facilities were not found in region_id={region_id}: {missing_text}"
        )

    return resolved


def upsert_user(cur, record: DQOUserRecord, region_id: int) -> tuple[int, str]:
    password_hash = bcrypt.hashpw(
        record.password.encode("utf-8"), bcrypt.gensalt()
    ).decode("utf-8")

    cur.execute("SELECT user_id FROM users WHERE username = %s", (record.username,))
    row = cur.fetchone()

    if row:
        user_id = int(row[0])
        cur.execute(
            """
            UPDATE users
            SET password_hash = %s,
                first_name = %s,
                last_name = %s,
                role = %s,
                facility_id = NULL,
                region_id = %s,
                country_id = NULL
            WHERE user_id = %s
            """,
            (
                password_hash,
                record.first_name,
                record.last_name,
                DQ_OFFICER_ROLE,
                region_id,
                user_id,
            ),
        )
        return user_id, "updated"

    cur.execute(
        """
        INSERT INTO users (
            username,
            password_hash,
            first_name,
            last_name,
            role,
            facility_id,
            region_id,
            country_id
        )
        VALUES (%s, %s, %s, %s, %s, NULL, %s, NULL)
        RETURNING user_id
        """,
        (
            record.username,
            password_hash,
            record.first_name,
            record.last_name,
            DQ_OFFICER_ROLE,
            region_id,
        ),
    )
    user_id = int(cur.fetchone()[0])
    return user_id, "created"


def sync_user_facility_access(cur, user_id: int, facility_ids: list[int]) -> None:
    cur.execute("DELETE FROM user_facility_access WHERE user_id = %s", (user_id,))
    cur.executemany(
        "INSERT INTO user_facility_access (user_id, facility_id) VALUES (%s, %s)",
        [(user_id, facility_id) for facility_id in facility_ids],
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Import DQ officer users from an Excel workbook into PostgreSQL."
    )
    parser.add_argument(
        "--xlsx",
        default=str(DEFAULT_WORKBOOK),
        help="Path to the DQO workbook. Defaults to utils/DQO_users.xlsx",
    )
    parser.add_argument(
        "--username",
        help="Optional username filter if you only want to sync one workbook user.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview the changes and roll the transaction back instead of committing.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / ".env")

    workbook_path = Path(args.xlsx).expanduser().resolve()
    if not workbook_path.exists():
        raise FileNotFoundError(f"Workbook not found: {workbook_path}")

    records = load_records(workbook_path, args.username)
    print(f"Loaded {len(records)} DQO user definition(s) from {workbook_path}")

    conn = get_db_connection()
    summaries: list[str] = []

    try:
        cur = conn.cursor()
        ensure_schema(cur)

        for record in records:
            region_id, matched_region_name = resolve_region(cur, record.region_name)
            matched_facilities = resolve_facilities(cur, region_id, record.facility_names)
            facility_ids = [facility_id for facility_id, _ in matched_facilities]
            user_id, action = upsert_user(cur, record, region_id)
            sync_user_facility_access(cur, user_id, facility_ids)

            summaries.append(
                (
                    f"{action.upper()}: username={record.username}, role={DQ_OFFICER_ROLE}, "
                    f"region={matched_region_name}, facilities={len(matched_facilities)}, user_id={user_id}"
                )
            )

        if args.dry_run:
            conn.rollback()
            print("Dry run complete. No database changes were committed.")
        else:
            conn.commit()
            print("Changes committed successfully.")

        for line in summaries:
            print(line)

    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
