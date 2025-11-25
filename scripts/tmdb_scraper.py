from __future__ import annotations
import argparse
import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

BASE_URL = "https://api.themoviedb.org/3"


# ---------- database helpers ----------

def get_conn(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.executescript(
        """
        CREATE TABLE IF NOT EXISTS movies (
            id INTEGER PRIMARY KEY,
            tmdb_id INTEGER UNIQUE,
            title TEXT,
            original_title TEXT,
            overview TEXT,
            release_date TEXT,
            runtime INTEGER,
            vote_average REAL,
            vote_count INTEGER,
            popularity REAL,
            original_language TEXT
        );

        CREATE TABLE IF NOT EXISTS genres (
            id INTEGER PRIMARY KEY,
            tmdb_id INTEGER UNIQUE,
            name TEXT
        );

        CREATE TABLE IF NOT EXISTS movie_genres (
            movie_id INTEGER,
            genre_id INTEGER,
            UNIQUE(movie_id, genre_id),
            FOREIGN KEY(movie_id) REFERENCES movies(id) ON DELETE CASCADE,
            FOREIGN KEY(genre_id) REFERENCES genres(id) ON DELETE CASCADE
        );
        """
    )
    conn.commit()


def get_or_create(conn: sqlite3.Connection, table: str, tmdb_id: int, name: str) -> int:
    """Return local row id for a TMDB id, inserting if needed."""
    cur = conn.execute(f"SELECT id FROM {table} WHERE tmdb_id = ?", (tmdb_id,))
    row = cur.fetchone()
    if row:
        return row[0]

    cur = conn.execute(
        f"INSERT INTO {table} (tmdb_id, name) VALUES (?, ?)",
        (tmdb_id, name),
    )
    return cur.lastrowid


# ---------- TMDB API helpers ----------

def get_api_key() -> str:
    key = os.getenv("TMDB_API_KEY")
    if not key:
        print("ERROR: TMDB_API_KEY not set", file=sys.stderr)
        sys.exit(1)
    return key


def fetch_movie(tmdb_id: int, retries: int = 3) -> Dict[str, Any]:
    """Fetch movie details (including keywords)."""
    api_key = get_api_key()
    params = {
        "api_key": api_key,
    }

    url = f"{BASE_URL}/movie/{tmdb_id}"
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=15)
            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", "2"))
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            print(f"Request failed for {tmdb_id}: {e}", file=sys.stderr)
            time.sleep(1.0)
    raise RuntimeError(f"Failed to fetch TMDB id {tmdb_id} after {retries} attempts")


# ---------- persistence ----------

def save_movie(conn: sqlite3.Connection, payload: Dict[str, Any]) -> None:
    """Save a single TMDB movie JSON into the database."""
    tmdb_id = int(payload["id"])
    cur = conn.cursor()

    # upsert movie row
    cur.execute(
        """
        INSERT INTO movies (
            tmdb_id, title, original_title, overview, release_date,
            runtime, vote_average, vote_count, popularity, original_language
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(tmdb_id) DO UPDATE SET
            title = excluded.title,
            original_title = excluded.original_title,
            overview = excluded.overview,
            release_date = excluded.release_date,
            runtime = excluded.runtime,
            vote_average = excluded.vote_average,
            vote_count = excluded.vote_count,
            popularity = excluded.popularity,
            original_language = excluded.original_language
        """,
        (
            tmdb_id,
            payload.get("title"),
            payload.get("original_title"),
            payload.get("overview"),
            payload.get("release_date"),
            payload.get("runtime"),
            payload.get("vote_average"),
            payload.get("vote_count"),
            payload.get("popularity"),
            payload.get("original_language"),
        ),
    )

    # local movie id
    cur.execute("SELECT id FROM movies WHERE tmdb_id = ?", (tmdb_id,))
    movie_rowid = cur.fetchone()[0]

    # genres
    cur.execute("DELETE FROM movie_genres WHERE movie_id = ?", (movie_rowid,))
    genre_rows: list[tuple[int, int]] = []
    for g in payload.get("genres", []):
        gid = get_or_create(conn, "genres", g["id"], g.get("name", ""))
        genre_rows.append((movie_rowid, gid))
    cur.executemany(
        "INSERT OR IGNORE INTO movie_genres (movie_id, genre_id) VALUES (?, ?)",
        genre_rows,
    )

    conn.commit()

# ---------- CLI ----------

def main() -> int:
    ap = argparse.ArgumentParser(description="Simple TMDB scraper.")
    ap.add_argument("--ids-file", type=Path, required=True,
                    help="Text file with one TMDB id per line")
    ap.add_argument("--db", default="tmdb.sqlite",
                    help="SQLite database path (default: tmdb.sqlite)")
    args = ap.parse_args()

    ids: list[int] = []
    with args.ids_file.open() as f:
        for line in f:
            line = line.strip()
            if line.isdigit():
                ids.append(int(line))

    if not ids:
        print("No valid ids in file!", file=sys.stderr)
        return 1

    conn = get_conn(args.db)
    init_db(conn)

    for tmdb_id in ids:
        try:
            payload = fetch_movie(tmdb_id)
            save_movie(conn, payload)
            print(f"✓ saved {payload.get('title')} (TMDB {tmdb_id})")
            time.sleep(0.3)
        except Exception as e:
            print(f"✗ failed {tmdb_id}: {e}", file=sys.stderr)

    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())