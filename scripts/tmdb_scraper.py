from __future__ import annotations
import argparse
import os
import sqlite3
import sys
import time
from collections import deque
from pathlib import Path
from typing import Dict, Any, List, Tuple

import asyncio
import aiohttp

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
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
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


# Simple async rate limiter: max_calls per period seconds (sliding window)
class RateLimiter:
    def __init__(self, max_calls: int, period: float) -> None:
        self.max_calls = max_calls
        self.period = period
        self._calls: deque[float] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            loop = asyncio.get_running_loop()
            now = loop.time()

            # Drop old entries
            while self._calls and self._calls[0] <= now - self.period:
                self._calls.popleft()

            if len(self._calls) >= self.max_calls:
                wait_for = self.period - (now - self._calls[0])
                await asyncio.sleep(wait_for)
                # After sleeping, call again to recompute window
                return await self.acquire()

            self._calls.append(now)


async def fetch_movie_async(
    session: aiohttp.ClientSession,
    tmdb_id: int,
    api_key: str,
    limiter: RateLimiter,
    retries: int = 3,
) -> Dict[str, Any]:
    """Async fetch movie details with retry and 429 handling."""
    params = {"api_key": api_key}
    url = f"{BASE_URL}/movie/{tmdb_id}"

    for attempt in range(retries):
        try:
            await limiter.acquire()
            async with session.get(url, params=params, timeout=15) as resp:
                if resp.status == 429:
                    retry_after = resp.headers.get("Retry-After")
                    wait = int(retry_after) if retry_after and retry_after.isdigit() else 2
                    print(f"429 for {tmdb_id}, retrying in {wait}s...", file=sys.stderr)
                    await asyncio.sleep(wait)
                    continue

                resp.raise_for_status()
                return await resp.json()

        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            print(f"Request failed for {tmdb_id} (attempt {attempt+1}/{retries}): {e}",
                  file=sys.stderr)
            await asyncio.sleep(1.0)

    raise RuntimeError(f"Failed to fetch TMDB id {tmdb_id} after {retries} attempts")


# ---------- persistence ----------

def save_movie(conn: sqlite3.Connection, payload: Dict[str, Any]) -> None:
    """Save a single TMDB movie JSON into the database."""
    tmdb_id = int(payload["id"])
    cur = conn.cursor()

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


# ---------- async orchestration ----------

async def fetch_all_movies(ids: List[int]) -> List[Dict[str, Any]]:
    api_key = get_api_key()
    limiter = RateLimiter(max_calls=50, period=1.0)  # 50 requests / second
    semaphore = asyncio.Semaphore(20)  # max 20 concurrent connections

    results: List[Dict[str, Any]] = []

    async with aiohttp.ClientSession() as session:
        async def worker(tmdb_id: int) -> Tuple[int, Dict[str, Any] | None]:
            try:
                async with semaphore:
                    payload = await fetch_movie_async(
                        session=session,
                        tmdb_id=tmdb_id,
                        api_key=api_key,
                        limiter=limiter,
                    )
                return tmdb_id, payload
            except Exception as e:
                print(f"✗ failed {tmdb_id}: {e}", file=sys.stderr)
                return tmdb_id, None

        tasks = [asyncio.create_task(worker(tmdb_id)) for tmdb_id in ids]

        # Use as_completed to get results as they arrive
        completed = 0
        total = len(tasks)
        for coro in asyncio.as_completed(tasks):
            tmdb_id, payload = await coro
            completed += 1
            if payload is not None:
                results.append(payload)
                title = payload.get("title") or "(no title)"
                print(f"[{completed}/{total}] ✓ fetched {title} (TMDB {tmdb_id})")
            else:
                print(f"[{completed}/{total}] ✗ no payload for TMDB {tmdb_id}",
                      file=sys.stderr)

    return results


# ---------- CLI ----------

def main() -> int:
    ap = argparse.ArgumentParser(description="Fast TMDB scraper with async + rate limiting.")
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

    # 1) Fetch all movies via async HTTP, respecting rate limits
    start_time = time.time()
    payloads = asyncio.run(fetch_all_movies(ids))
    fetch_duration = time.time() - start_time
    print(f"\nFetched {len(payloads)} movies in {fetch_duration:.1f} seconds.")

    # 2) Store them in SQLite (single-threaded, batched transaction)
    conn = get_conn(args.db)
    init_db(conn)

    cur = conn.cursor()
    cur.execute("BEGIN")
    try:
        for idx, payload in enumerate(payloads, start=1):
            save_movie(conn, payload)
            title = payload.get("title") or "(no title)"
            print(f"[DB {idx}/{len(payloads)}] saved {title} (TMDB {payload['id']})")
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    total_duration = time.time() - start_time
    print(f"\nAll done in {total_duration:.1f} seconds.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())