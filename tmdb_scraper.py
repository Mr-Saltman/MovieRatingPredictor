#!/usr/bin/env python3
"""
TMDB Movie Scraper — optimized for speed & cleanliness
-----------------------------------------------------

Key improvements vs. earlier version
- **Fewer commits**: transactions are batched (per page / per movie), not per row.
- **WAL mode** + sensible PRAGMAs for faster local writes.
- **Entity ID caches** (genres/people/keywords/companies) to avoid redundant SELECTs.
- **Bulk inserts** (executemany) for cast/crew/keywords/link tables.
- **Insert-only fast path** (`--insert-only`) optionally **skips API fetch** when a movie already exists.
- **Parent parser** so common flags work before/after subcommands.
- **Defensive rate limit handling** and exponential backoff.
- **No API key in code**; reads from `TMDB_API_KEY` (optionally via `.env`).

Usage examples
  export TMDB_API_KEY=...    # or use a .env file (python-dotenv)
  python tmdb_scraper.py popular --pages 5 --db tmdb.sqlite
  python tmdb_scraper.py trending --window week --pages 2 --insert-only
  python tmdb_scraper.py movie 603 27205 --db data/movies.sqlite
  python tmdb_scraper.py from-file ids.txt --insert-only

"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import Iterable, Optional, Dict, Tuple, List

import requests

# Optional: load .env if available
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

BASE_URL = "https://api.themoviedb.org/3"
SESSION = requests.Session()
SESSION.headers.update({"Accept": "application/json"})

# ------------------------------
# Database setup & helpers
# ------------------------------

def get_conn(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    # Performance-oriented PRAGMAs (safe-ish for local dev; adjust for prod)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA temp_store=MEMORY;")
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
            original_language TEXT,
            poster_path TEXT,
            backdrop_path TEXT,
            homepage TEXT,
            status TEXT
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

        CREATE TABLE IF NOT EXISTS people (
            id INTEGER PRIMARY KEY,
            tmdb_id INTEGER UNIQUE,
            name TEXT
        );

        CREATE TABLE IF NOT EXISTS cast (
            movie_id INTEGER,
            person_id INTEGER,
            character TEXT,
            cast_order INTEGER,
            UNIQUE(movie_id, person_id, character),
            FOREIGN KEY(movie_id) REFERENCES movies(id) ON DELETE CASCADE,
            FOREIGN KEY(person_id) REFERENCES people(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS crew (
            movie_id INTEGER,
            person_id INTEGER,
            department TEXT,
            job TEXT,
            UNIQUE(movie_id, person_id, job),
            FOREIGN KEY(movie_id) REFERENCES movies(id) ON DELETE CASCADE,
            FOREIGN KEY(person_id) REFERENCES people(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS keywords (
            id INTEGER PRIMARY KEY,
            tmdb_id INTEGER UNIQUE,
            name TEXT
        );

        CREATE TABLE IF NOT EXISTS movie_keywords (
            movie_id INTEGER,
            keyword_id INTEGER,
            UNIQUE(movie_id, keyword_id),
            FOREIGN KEY(movie_id) REFERENCES movies(id) ON DELETE CASCADE,
            FOREIGN KEY(keyword_id) REFERENCES keywords(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS companies (
            id INTEGER PRIMARY KEY,
            tmdb_id INTEGER UNIQUE,
            name TEXT
        );

        CREATE TABLE IF NOT EXISTS movie_companies (
            movie_id INTEGER,
            company_id INTEGER,
            UNIQUE(movie_id, company_id),
            FOREIGN KEY(movie_id) REFERENCES movies(id) ON DELETE CASCADE,
            FOREIGN KEY(company_id) REFERENCES companies(id) ON DELETE CASCADE
        );
        """
    )
    conn.commit()


# ------------------------------
# ID caches to reduce DB hits
# ------------------------------
class IdCache:
    def __init__(self, table: str):
        self.table = table
        self.map: Dict[int, int] = {}

    def get(self, tmdb_id: int) -> Optional[int]:
        return self.map.get(tmdb_id)

    def remember(self, tmdb_id: int, row_id: int) -> None:
        self.map[tmdb_id] = row_id


genre_cache = IdCache("genres")
people_cache = IdCache("people")
keyword_cache = IdCache("keywords")
company_cache = IdCache("companies")
movie_cache = IdCache("movies")


# ------------------------------
# Upsert helpers (no per-row commit; caller manages transactions)
# ------------------------------

def _get_id(conn: sqlite3.Connection, table: str, tmdb_id: int) -> Optional[int]:
    cur = conn.execute(f"SELECT id FROM {table} WHERE tmdb_id=?", (tmdb_id,))
    row = cur.fetchone()
    return row[0] if row else None


def _insert_or_update(conn: sqlite3.Connection, table: str, tmdb_id: int, name: str) -> int:
    cur = conn.cursor()
    cur.execute(
        f"INSERT INTO {table}(tmdb_id, name) VALUES(?, ?) ON CONFLICT(tmdb_id) DO UPDATE SET name=excluded.name",
        (tmdb_id, name),
    )
    row_id = _get_id(conn, table, tmdb_id)
    return int(row_id)


def upsert_genre(conn: sqlite3.Connection, tmdb_id: int, name: str) -> int:
    cached = genre_cache.get(tmdb_id)
    if cached is not None:
        return cached
    row_id = _insert_or_update(conn, "genres", tmdb_id, name)
    genre_cache.remember(tmdb_id, row_id)
    return row_id


def upsert_person(conn: sqlite3.Connection, tmdb_id: int, name: str) -> int:
    cached = people_cache.get(tmdb_id)
    if cached is not None:
        return cached
    row_id = _insert_or_update(conn, "people", tmdb_id, name)
    people_cache.remember(tmdb_id, row_id)
    return row_id


def upsert_keyword(conn: sqlite3.Connection, tmdb_id: int, name: str) -> int:
    cached = keyword_cache.get(tmdb_id)
    if cached is not None:
        return cached
    row_id = _insert_or_update(conn, "keywords", tmdb_id, name)
    keyword_cache.remember(tmdb_id, row_id)
    return row_id


def upsert_company(conn: sqlite3.Connection, tmdb_id: int, name: str) -> int:
    cached = company_cache.get(tmdb_id)
    if cached is not None:
        return cached
    row_id = _insert_or_update(conn, "companies", tmdb_id, name)
    company_cache.remember(tmdb_id, row_id)
    return row_id


def upsert_movie(conn: sqlite3.Connection, m: dict, *, update: bool = True) -> int:
    cached = movie_cache.get(int(m.get("id")))
    if cached is not None:
        return cached

    cur = conn.cursor()
    if update:
        cur.execute(
            """
            INSERT INTO movies (
                tmdb_id, title, original_title, overview, release_date, runtime,
                vote_average, vote_count, popularity, original_language, poster_path,
                backdrop_path, homepage, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(tmdb_id) DO UPDATE SET
                title=excluded.title,
                original_title=excluded.original_title,
                overview=excluded.overview,
                release_date=excluded.release_date,
                runtime=excluded.runtime,
                vote_average=excluded.vote_average,
                vote_count=excluded.vote_count,
                popularity=excluded.popularity,
                original_language=excluded.original_language,
                poster_path=excluded.poster_path,
                backdrop_path=excluded.backdrop_path,
                homepage=excluded.homepage,
                status=excluded.status
            """,
            (
                m.get("id"), m.get("title"), m.get("original_title"), m.get("overview"),
                m.get("release_date"), m.get("runtime"), m.get("vote_average"),
                m.get("vote_count"), m.get("popularity"), m.get("original_language"),
                m.get("poster_path"), m.get("backdrop_path"), m.get("homepage"),
                m.get("status"),
            ),
        )
    else:
        cur.execute(
            """
            INSERT OR IGNORE INTO movies (
                tmdb_id, title, original_title, overview, release_date, runtime,
                vote_average, vote_count, popularity, original_language, poster_path,
                backdrop_path, homepage, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                m.get("id"), m.get("title"), m.get("original_title"), m.get("overview"),
                m.get("release_date"), m.get("runtime"), m.get("vote_average"),
                m.get("vote_count"), m.get("popularity"), m.get("original_language"),
                m.get("poster_path"), m.get("backdrop_path"), m.get("homepage"),
                m.get("status"),
            ),
        )

    row_id = _get_id(conn, "movies", int(m.get("id")))
    movie_cache.remember(int(m.get("id")), int(row_id))
    return int(row_id)


# ------------------------------
# TMDB API helpers
# ------------------------------

def get_api_key() -> str:
    key = os.getenv("TMDB_API_KEY")
    if not key:
        print("ERROR: TMDB_API_KEY not set. Export it or put it in a .env file.", file=sys.stderr)
        sys.exit(1)
    return key


def fetch(path: str, params: Optional[dict] = None, *, retries: int = 3, base_sleep: float = 0.35) -> dict:
    """Fetch JSON from TMDB with polite delay and exponential backoff."""
    params = dict(params or {})
    params["api_key"] = get_api_key()

    sleep = base_sleep
    for attempt in range(1, retries + 1):
        try:
            resp = SESSION.get(f"{BASE_URL}{path}", params=params, timeout=20)
            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", "2"))
                time.sleep(wait)
                continue
            resp.raise_for_status()
            time.sleep(base_sleep)
            return resp.json()
        except requests.RequestException:
            if attempt == retries:
                raise
            time.sleep(sleep)
            sleep *= 2
    return {}


def fetch_movie_full(tmdb_id: int) -> dict:
    return fetch(
        f"/movie/{tmdb_id}",
        params={"append_to_response": "credits,keywords"},
    )


def paged(endpoint: str, *, pages: int = 1, params: Optional[dict] = None) -> Iterable[dict]:
    params = dict(params or {})
    for page in range(1, pages + 1):
        params.update({"page": page})
        data = fetch(endpoint, params=params)
        for r in data.get("results", []) or []:
            yield r


# ------------------------------
# Persistence logic for a full movie payload
# ------------------------------

def save_movie_payload(conn: sqlite3.Connection, payload: dict, *, insert_only: bool = False) -> int:
    # One transaction per movie to cut fsync overhead
    with conn:
        mid = upsert_movie(conn, payload, update=not insert_only)

        # Genres
        genre_rows: List[Tuple[int, int]] = []
        for g in payload.get("genres", []) or []:
            gid = upsert_genre(conn, g["id"], g.get("name", ""))
            genre_rows.append((mid, gid))
        if genre_rows:
            conn.executemany(
                "INSERT OR IGNORE INTO movie_genres(movie_id, genre_id) VALUES (?, ?)",
                genre_rows,
            )

        # Production companies
        comp_rows: List[Tuple[int, int]] = []
        for c in payload.get("production_companies", []) or []:
            cid = upsert_company(conn, c["id"], c.get("name", ""))
            comp_rows.append((mid, cid))
        if comp_rows:
            conn.executemany(
                "INSERT OR IGNORE INTO movie_companies(movie_id, company_id) VALUES (?, ?)",
                comp_rows,
            )

        # Keywords (payload["keywords"] may be a dict with key "keywords")
        kw_container = payload.get("keywords") or {}
        kw_list = kw_container.get("keywords") if isinstance(kw_container, dict) else kw_container
        kw_rows: List[Tuple[int, int]] = []
        for kw in kw_list or []:
            kwid = upsert_keyword(conn, kw["id"], kw.get("name", ""))
            kw_rows.append((mid, kwid))
        if kw_rows:
            conn.executemany(
                "INSERT OR IGNORE INTO movie_keywords(movie_id, keyword_id) VALUES (?, ?)",
                kw_rows,
            )

        # Credits
        credits = payload.get("credits") or {}
        cast_rows: List[Tuple[int, int, Optional[str], Optional[int]]] = []
        for c in credits.get("cast", []) or []:
            pid = upsert_person(conn, c.get("id"), c.get("name", ""))
            cast_rows.append((mid, pid, c.get("character"), c.get("order")))
        if cast_rows:
            conn.executemany(
                "INSERT OR IGNORE INTO cast(movie_id, person_id, character, cast_order) VALUES (?, ?, ?, ?)",
                cast_rows,
            )

        crew_rows: List[Tuple[int, int, Optional[str], Optional[str]]] = []
        for c in credits.get("crew", []) or []:
            pid = upsert_person(conn, c.get("id"), c.get("name", ""))
            crew_rows.append((mid, pid, c.get("department"), c.get("job")))
        if crew_rows:
            conn.executemany(
                "INSERT OR IGNORE INTO crew(movie_id, person_id, department, job) VALUES (?, ?, ?, ?)",
                crew_rows,
            )

    return mid


# ------------------------------
# Command handlers
# ------------------------------

def movie_exists(conn: sqlite3.Connection, tmdb_id: int) -> bool:
    cached = movie_cache.get(tmdb_id)
    if cached is not None:
        return True
    cur = conn.execute("SELECT 1 FROM movies WHERE tmdb_id=?", (tmdb_id,))
    return cur.fetchone() is not None


def handle_movies_from_iterable(conn: sqlite3.Connection, ids: Iterable[int], *, insert_only: bool = False, skip_existing_fetch: bool = True) -> None:
    for tmdb_id in ids:
        try:
            if insert_only and skip_existing_fetch and movie_exists(conn, int(tmdb_id)):
                # Fast path: avoid API call entirely
                continue
            payload = fetch_movie_full(int(tmdb_id))
            save_movie_payload(conn, payload, insert_only=insert_only)
            print(f"Saved movie {payload.get('title')} (TMDB {tmdb_id})")
        except Exception as e:
            print(f"Failed TMDB id {tmdb_id}: {e}", file=sys.stderr)


def cmd_movie(conn: sqlite3.Connection, args: argparse.Namespace) -> None:
    handle_movies_from_iterable(
        conn,
        (int(x) for x in args.ids),
        insert_only=args.insert_only,
        skip_existing_fetch=not args.no_skip_existing,
    )


def cmd_from_file(conn: sqlite3.Connection, args: argparse.Namespace) -> None:
    path = Path(args.path)
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(2)
    with path.open() as f:
        ids = [int(line.strip()) for line in f if line.strip() and line.strip().isdigit()]
    handle_movies_from_iterable(
        conn,
        ids,
        insert_only=args.insert_only,
        skip_existing_fetch=not args.no_skip_existing,
    )


def _expand_and_save_movie(conn: sqlite3.Connection, brief: dict, *, insert_only: bool, skip_existing_fetch: bool) -> None:
    tmdb_id = brief.get("id")
    if tmdb_id is None:
        return
    if insert_only and skip_existing_fetch and movie_exists(conn, int(tmdb_id)):
        return
    payload = fetch_movie_full(int(tmdb_id))
    save_movie_payload(conn, payload, insert_only=insert_only)
    print(f"Saved movie {payload.get('title')} (TMDB {tmdb_id})")


def cmd_popular(conn: sqlite3.Connection, args: argparse.Namespace) -> None:
    # Batch a page per transaction implicitly inside save_movie_payload
    for brief in paged("/movie/popular", pages=args.pages, params={"language": args.language}):
        _expand_and_save_movie(conn, brief, insert_only=args.insert_only, skip_existing_fetch=not args.no_skip_existing)


def cmd_top_rated(conn: sqlite3.Connection, args: argparse.Namespace) -> None:
    for brief in paged("/movie/top_rated", pages=args.pages, params={"language": args.language}):
        _expand_and_save_movie(conn, brief, insert_only=args.insert_only, skip_existing_fetch=not args.no_skip_existing)


def cmd_trending(conn: sqlite3.Connection, args: argparse.Namespace) -> None:
    window = args.window  # day or week
    for brief in paged(f"/trending/movie/{window}", pages=args.pages, params={"language": args.language}):
        _expand_and_save_movie(conn, brief, insert_only=args.insert_only, skip_existing_fetch=not args.no_skip_existing)


# ------------------------------
# CLI wiring
# ------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Scrape TMDB movies into SQLite without duplicates.")

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--db", default="tmdb.sqlite", help="SQLite file path (default: tmdb.sqlite)")
    common.add_argument("--language", default="en-US", help="TMDB language parameter, e.g., en-US")
    common.add_argument("--insert-only", action="store_true", help="Skip updates to existing movies (faster).")
    common.add_argument("--no-skip-existing", dest="no_skip_existing", action="store_true",
                        help="When --insert-only is set, still fetch details even if movie exists.")

    sub = p.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("popular", parents=[common], help="Fetch popular movies")
    ps.add_argument("--pages", type=int, default=1, help="Number of pages to fetch")

    ts = sub.add_parser("top_rated", parents=[common], help="Fetch top rated movies")
    ts.add_argument("--pages", type=int, default=1, help="Number of pages to fetch")

    tr = sub.add_parser("trending", parents=[common], help="Fetch trending movies")
    tr.add_argument("--window", choices=["day", "week"], default="day", help="Trending window")
    tr.add_argument("--pages", type=int, default=1, help="Number of pages to fetch")

    mv = sub.add_parser("movie", parents=[common], help="Fetch explicit TMDB movie IDs")
    mv.add_argument("ids", nargs="+", help="One or more TMDB movie IDs")

    ff = sub.add_parser("from-file", parents=[common], help="Fetch movie IDs from a text file (one TMDB id per line)")
    ff.add_argument("path", help="Path to file containing TMDB IDs")

    return p


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    conn = get_conn(args.db)
    init_db(conn)

    if args.cmd == "popular":
        cmd_popular(conn, args)
    elif args.cmd == "top_rated":
        cmd_top_rated(conn, args)
    elif args.cmd == "trending":
        cmd_trending(conn, args)
    elif args.cmd == "movie":
        cmd_movie(conn, args)
    elif args.cmd == "from-file":
        cmd_from_file(conn, args)
    else:
        print("Unknown command", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
