"""Fetch random YouTube thumbnails across niches for research use (PRD section 8).

Searches with yt-dlp (metadata only, no video is downloaded), then downloads maxresdefault.jpg
from i.ytimg.com. Only true 1280x720 thumbnails are kept. One thumbnail per channel, so the set is varied
and a channel-level train/test split is possible later (see manifest.csv).

Run from backend/:
    .venv\\Scripts\\python scripts\\fetch_thumbnails.py                 # 24 thumbnails, 3 per niche
    .venv\\Scripts\\python scripts\\fetch_thumbnails.py --per-niche 6   # 48
    .venv\\Scripts\\python scripts\\fetch_thumbnails.py --seed 7        # reproducible pick
    .venv\\Scripts\\python scripts\\fetch_thumbnails.py --dry-run       # search only, download nothing

Re-running adds new ones and skips ids already in data/thumbnails/manifest.csv.
Thumbnails are for research use only; do not redistribute them.
"""
from __future__ import annotations

import argparse
import csv
import io
import random
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image

OUT_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "thumbnails"
MANIFEST = OUT_DIR / "manifest.csv"
FIELDS = ["id", "niche", "channel_id", "channel", "title", "query", "file"]

NICHES = {
    "gaming": ["minecraft survival challenge", "fortnite funny moments", "gta 5 mods", "horror game reaction"],
    "tech": ["smartphone review 2025", "laptop vs macbook comparison", "pc build guide", "gadget unboxing"],
    "vlog": ["day in my life vlog", "moving to a new city vlog", "family vlog", "travel vlog"],
    "education": ["how does it work explained", "science experiment explained", "history documentary short", "learn python tutorial"],
    "finance": ["how to invest for beginners", "side hustle ideas", "stock market crash explained", "make money online"],
    "fitness": ["home workout no equipment", "weight loss transformation", "gym routine for beginners", "yoga for beginners"],
    "food": ["easy dinner recipe", "street food tour", "baking cake recipe", "trying viral food"],
    "entertainment": ["reaction video", "mrbeast style challenge", "prank gone wrong", "celebrity interview funny"],
}
MIN_DURATION_S = 90  # skip Shorts
UA = {"User-Agent": "Mozilla/5.0 (research thumbnail fetch)"}


def search(query: str, n: int) -> list[dict]:
    import yt_dlp

    opts = {"quiet": True, "no_warnings": True, "extract_flat": True, "skip_download": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(f"ytsearch{n}:{query}", download=False)
    return [e for e in (info or {}).get("entries", []) if e and e.get("id")]


def fetch_maxres(video_id: str) -> bytes | None:
    url = f"https://i.ytimg.com/vi/{video_id}/maxresdefault.jpg"
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=15) as r:
            data = r.read()
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
        return None
    try:
        with Image.open(io.BytesIO(data)) as im:
            if im.size != (1280, 720):  # 120x90 placeholders and odd sizes are rejected
                return None
    except Exception:
        return None
    return data


def load_manifest() -> dict[str, dict]:
    if not MANIFEST.exists():
        return {}
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        return {row["id"]: row for row in csv.DictReader(f)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-niche", type=int, default=3)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true", help="search and list picks, download nothing")
    args = ap.parse_args()

    try:
        import yt_dlp  # noqa: F401
    except ImportError:
        print("yt-dlp is missing. Run:  .venv\\Scripts\\python -m pip install yt-dlp")
        return 1

    rng = random.Random(args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seen = load_manifest()
    seen_channels = {r["channel_id"] for r in seen.values() if r["channel_id"]}
    new_rows: list[dict] = []

    for niche, queries in NICHES.items():
        query = rng.choice(queries)
        print(f"[{niche}] searching: {query!r}")
        try:
            results = search(query, 30)
        except Exception as e:
            print(f"  search failed: {str(e)[:120]}")
            continue
        rng.shuffle(results)
        got = 0
        for e in results:
            if got >= args.per_niche:
                break
            vid, ch = e["id"], e.get("channel_id") or e.get("uploader_id") or ""
            if vid in seen or (ch and ch in seen_channels):
                continue
            if (e.get("duration") or MIN_DURATION_S) < MIN_DURATION_S:
                continue
            if args.dry_run:
                print(f"  would fetch {vid}  {e.get('channel', '')[:25]:25}  {e.get('title', '')[:50]}")
                got += 1
                continue
            data = fetch_maxres(vid)
            time.sleep(0.3)
            if data is None:
                continue
            fname = f"{niche}_{vid}.jpg"
            (OUT_DIR / fname).write_bytes(data)
            row = {"id": vid, "niche": niche, "channel_id": ch, "channel": e.get("channel") or e.get("uploader") or "",
                   "title": e.get("title", ""), "query": query, "file": fname}
            new_rows.append(row)
            seen[vid] = row
            seen_channels.add(ch)
            got += 1
            print(f"  saved {fname}  ({row['channel'][:30]})")
        if got < args.per_niche and not args.dry_run:
            print(f"  only got {got}/{args.per_niche} (re-run to top up)")

    if new_rows:
        write_header = not MANIFEST.exists()
        with open(MANIFEST, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if write_header:
                w.writeheader()
            w.writerows(new_rows)
    total = len(list(OUT_DIR.glob("*.jpg")))
    print(f"\nDone. {len(new_rows)} new, {total} total in {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
