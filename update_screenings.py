"""Weekly NYC repertory screening updater.

Fetches showtimes for tracked art-house venues via SerpApi, enriches each
film with TMDB metadata, scores it against taste_profile.json, and writes
screenings.json consumed by index.html.

Required env: SERPAPI_API_KEY
Optional env: TMDB_API_KEY (enrichment + scoring degrade gracefully without)
              SERPAPI_DEBUG=1 (dumps raw SerpApi responses to
              debug_serpapi.json for parser diagnosis)
"""
import datetime
import html
import json
import os
import re
import sys
import zlib
from collections import defaultdict

import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from tmdb_client import TMDBClient, clean_film_title, trim_summary

# ---------------------------------------------------------------------------
# 1. Configuration & target weekend
# ---------------------------------------------------------------------------
SERPAPI_API_KEY = os.environ.get("SERPAPI_API_KEY", "").strip()
TMDB_API_KEY = os.environ.get("TMDB_API_KEY", "").strip()
DEBUG_SERPAPI = os.environ.get("SERPAPI_DEBUG", "").lower() in ("1", "true", "yes")
PROFILE_JSON = "taste_profile.json"
OUTPUT_JSON = "screenings.json"

today = datetime.date.today()
days_until_friday = (4 - today.weekday()) % 7  # 0 when run on a Friday
friday_date = today + datetime.timedelta(days=days_until_friday)
saturday_date = friday_date + datetime.timedelta(days=1)
sunday_date = friday_date + datetime.timedelta(days=2)

fri_str = friday_date.strftime("%b %d")
weekend_label = (f"{friday_date.strftime('%b %d')} \u2013 "
                 f"{sunday_date.strftime('%b %d')}")
print(f"[Calendar] Targeting weekend: {weekend_label}")

THEATER_MAP = {
    "amc lincoln square": ("AMC Lincoln Square 13", "Upper West Side",
        "https://www.amctheatres.com/movie-theatres/new-york-city/amc-lincoln-square-13"),
    "lincoln square 13": ("AMC Lincoln Square 13", "Upper West Side",
        "https://www.amctheatres.com/movie-theatres/new-york-city/amc-lincoln-square-13"),
    "lincoln square": ("AMC Lincoln Square 13", "Upper West Side",
        "https://www.amctheatres.com/movie-theatres/new-york-city/amc-lincoln-square-13"),
    "film forum": ("Film Forum", "Greenwich Village",
        "https://filmforum.org/now_playing"),
    "ifc center": ("IFC Center", "West Village", "https://www.ifccenter.com/"),
    "metrograph": ("Metrograph", "Lower East Side", "https://metrograph.com/nyc/"),
    "paris theater": ("The Paris Theater", "Midtown",
        "https://www.paristheaternyc.com/"),
    "the paris": ("The Paris Theater", "Midtown",
        "https://www.paristheaternyc.com/"),
    "roxy cinema": ("The Roxy Cinema", "Tribeca",
        "https://www.roxycinematribeca.com/"),
    "film at lincoln center": ("Film at Lincoln Center", "Upper West Side",
        "https://www.filmlinc.org/now-playing/"),
    "walter reade": ("Film at Lincoln Center", "Upper West Side",
        "https://www.filmlinc.org/now-playing/"),
    "bam rose": ("BAM Rose Cinemas", "Brooklyn", "https://www.bam.org/film"),
    "bam": ("BAM Rose Cinemas", "Brooklyn", "https://www.bam.org/film"),
    "nitehawk": ("Nitehawk Cinema", "Brooklyn", "https://nitehawkcinema.com/"),
    "angelika": ("Angelika Film Center", "SoHo",
        "https://www.angelikafilmcenter.com/nyc"),
    "cinema village": ("Cinema Village", "Greenwich Village",
        "https://www.cinemavillage.com/"),
}

STYLE_TROPES = [
    "nocturnal", "existential", "slow-burn", "kinetic", "neon", "melancholic",
    "paranoia", "isolation", "atmospheric", "stylized", "underworld", "obsession",
    "noir", "crime", "surreal", "laconic", "nihilistic", "poetic",
]

# ---------------------------------------------------------------------------
# 2. Taste profile
# ---------------------------------------------------------------------------
if os.path.exists(PROFILE_JSON):
    with open(PROFILE_JSON, "r", encoding="utf-8") as f:
        profile = json.load(f)
    watched_titles = set(profile.get("watched_titles", []))
    director_affinity = profile.get("director_affinity", {})
    dp_affinity = profile.get("dp_affinity", {})
    positive_review_text = profile.get("positive_review_text", "")
else:
    watched_titles = set()
    director_affinity = {}
    dp_affinity = {}
    positive_review_text = ""

tmdb = TMDBClient(TMDB_API_KEY)
if not tmdb.configured:
    print("[Warning] TMDB_API_KEY not set: enrichment and scoring will be "
          "limited to SerpApi data.")

# ---------------------------------------------------------------------------
# 3. SerpApi ingestion with defensive parsing
# ---------------------------------------------------------------------------
TIME_RE = re.compile(r"\b\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)\b")
SKIP_TITLES = {"tickets", "directions", "website", "showtimes"}


def _as_list(value):
    return value if isinstance(value, list) else []


def iter_movie_blocks(data):
    """Yield (raw_theater_name, movie_dict) from a SerpApi response.

    Google showtimes come back in several shapes depending on the query, so
    this tries the known layouts in order: top-level "showtimes", then the
    knowledge-graph variants, then movies/local results.
    """
    blocks = []
    if isinstance(data.get("showtimes"), list):
        blocks = data["showtimes"]
    elif isinstance(data.get("knowledge_graph"), dict):
        kg = data["knowledge_graph"]
        blocks = kg.get("movies_results") or kg.get("theaters") or []
    if not blocks:
        blocks = data.get("movies_results") or data.get("local_results") or []

    for block in blocks:
        if not isinstance(block, dict):
            continue
        theater_raw = (block.get("theater_name") or block.get("name")
                       or block.get("title", ""))
        movies = _as_list(block.get("movies"))
        if not movies and any(k in block for k in ("showtimes", "times", "showtime")):
            movies = [block]  # the block itself describes one movie
        for movie in movies:
            if isinstance(movie, dict):
                yield theater_raw, movie


def extract_times(movie):
    """Return (times, format_label) from a SerpApi movie dict.

    times is a list of (day_or_None, time_string). Handles string times,
    {"time","type"} dicts, and day-grouped dicts.
    """
    raw = movie.get("showtimes") or movie.get("times") or movie.get("showtime")
    pairs = []
    if isinstance(raw, dict):
        # Possibly grouped by day: {"Friday": [...], ...}
        looks_grouped = any(isinstance(v, list) for v in raw.values())
        if looks_grouped:
            for day, lst in raw.items():
                for item in _as_list(lst):
                    pairs.append((str(day), item))
        else:
            pairs.append((None, raw))
    else:
        for item in _as_list(raw):
            pairs.append((None, item))

    times = []
    fmt = "Standard DCP"
    for day, item in pairs:
        if isinstance(item, str):
            time_str, ftype, item_day = item.strip(), "", None
        elif isinstance(item, dict):
            time_str = (item.get("time") or item.get("showtime") or "").strip()
            ftype = (item.get("type") or "").lower()
            item_day = item.get("day")
        else:
            continue
        if not time_str or not TIME_RE.search(time_str):
            continue
        if "70mm" in ftype:
            fmt = "70mm"
        elif "imax" in ftype:
            fmt = "IMAX"
        elif "35mm" in ftype:
            fmt = "35mm Print"
        times.append((day or item_day, TIME_RE.search(time_str).group(0)))
    return times, fmt


def match_theater(raw_name, query):
    haystacks = [(raw_name or "").lower(), (query or "").lower()]
    for key, venue in THEATER_MAP.items():
        if any(key in h for h in haystacks):
            return venue
    return None


def label_showtime(day, time_str):
    """Render a chip label, preferring SerpApi's day when present."""
    if day:
        return f"{day}: {time_str}"
    return f"Fri {fri_str}: {time_str}"


def fetch_serpapi_showtimes():
    screenings = defaultdict(lambda: {
        "theater": None, "neighborhood": None, "ticket_url": None,
        "summary": "", "format": "DCP", "showtimes": [],
    })
    debug_dump = {}
    total_times = 0
    zero_time_movies = 0

    search_queries = [
        "AMC Lincoln Square 13 showtimes",
        "IFC Center NYC showtimes",
        "Film Forum NYC showtimes",
        "Metrograph NYC showtimes",
        "Film at Lincoln Center showtimes",
        "The Paris Theater NYC showtimes",
        "Roxy Cinema Tribeca showtimes",
        "Angelika Film Center NYC showtimes",
        "Nitehawk Cinema Brooklyn showtimes",
        "BAM Rose Cinemas showtimes",
        "Cinema Village NYC showtimes",
    ]

    for q in search_queries:
        print(f"[SerpApi] Querying: '{q}'...")
        params = {
            "engine": "google",
            "q": q,
            "location": "New York, New York, United States",
            "hl": "en",
            "gl": "us",
            "api_key": SERPAPI_API_KEY,
        }
        try:
            resp = requests.get("https://serpapi.com/search.json",
                                params=params, timeout=60)
            data = resp.json()
        except Exception as exc:
            print(f"[SerpApi Error] Query '{q}' failed: {exc}")
            continue

        if DEBUG_SERPAPI:
            debug_dump[q] = data

        if "error" in data:
            print(f"[SerpApi API Error] '{q}': {data['error']}")
            continue

        n_movies = n_times = 0
        for theater_raw, movie in iter_movie_blocks(data):
            venue = match_theater(theater_raw, q)
            if not venue:
                continue
            raw_title = movie.get("name") or movie.get("title", "")
            clean_t = clean_film_title(raw_title)
            if len(clean_t) < 2 or clean_t.lower() in SKIP_TITLES:
                continue

            times, fmt = extract_times(movie)
            n_movies += 1
            n_times += len(times)
            if not times:
                zero_time_movies += 1

            labels = [label_showtime(day, t) for day, t in times[:6]]
            if not labels:
                labels = [f"Fri {fri_str}: check venue site"]

            t_name, neigh, t_url = venue
            key = (clean_t.lower(), t_name)
            entry = screenings[key]
            entry["title"] = clean_t
            entry["theater"] = t_name
            entry["neighborhood"] = neigh
            entry["ticket_url"] = t_url
            entry["format"] = fmt
            entry["summary"] = f"Playing at {t_name}."
            for label in labels:
                if label not in entry["showtimes"]:
                    entry["showtimes"].append(label)

        total_times += n_times
        print(f"  -> {n_movies} movies, {n_times} showtimes parsed")

    if DEBUG_SERPAPI:
        with open("debug_serpapi.json", "w", encoding="utf-8") as f:
            json.dump(debug_dump, f, indent=2, ensure_ascii=False)
        print("[Debug] Raw SerpApi responses written to debug_serpapi.json")

    print(f"[SerpApi] {len(screenings)} screenings, {total_times} showtimes "
          f"total ({zero_time_movies} movies with no parsable times).")
    return [create_entry(**data) for data in screenings.values()]


# ---------------------------------------------------------------------------
# 4. Taste scoring & poster SVG fallback
# ---------------------------------------------------------------------------
def calculate_taste_score(title, director, summary, tmdb_info=None):
    score = 50.0
    dir_clean = str(director).lower().strip()

    dir_score = 0.0
    for d, weight in director_affinity.items():
        if d in dir_clean or dir_clean in d:
            dir_score += weight * 3.5
    score += max(-14.0, min(14.0, dir_score))

    if tmdb_info:
        dp_score = sum(dp_affinity.get(dp, 0.0) * 2.5
                       for dp in tmdb_info.get("dps", []))
        score += max(-10.0, min(10.0, dp_score))

    screening_text = (f"{summary} "
                      f"{tmdb_info.get('corpus', '') if tmdb_info else ''}")
    try:
        if positive_review_text.strip():
            tfidf = TfidfVectorizer().fit_transform(
                [positive_review_text, screening_text])
            sim = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
            score += round(sim * 14)
    except Exception:
        pass

    trope_count = sum(1 for trope in STYLE_TROPES
                      if trope in screening_text.lower())
    score += min(round(trope_count * 2.5), 10)

    return max(30, min(int(score), 98))


def generate_poster_svg(title, director, year):
    h = zlib.crc32(title.encode("utf-8"))
    palettes = [
        {"bg": "#080507", "primary": "#ff2a4b", "secondary": "#e5a93c", "accent": "#00e5bc"},
        {"bg": "#04080e", "primary": "#00e5bc", "secondary": "#ff2a4b", "accent": "#f3ebd7"},
        {"bg": "#0c0608", "primary": "#e5a93c", "secondary": "#ff2a4b", "accent": "#8b93a6"},
        {"bg": "#06090c", "primary": "#ff2a4b", "secondary": "#00e5bc", "accent": "#e5a93c"},
    ]
    p = palettes[h % len(palettes)]
    clean_display = html.escape(title.upper())
    font_size = 11 if len(clean_display) > 22 else 14
    return (
        f'<svg viewBox="0 0 200 300" xmlns="http://www.w3.org/2000/svg">'
        f'<rect width="200" height="300" fill="{p["bg"]}"/>'
        f'<circle cx="100" cy="100" r="48" fill="{p["primary"]}" opacity="0.8"/>'
        f'<text x="100" y="238" font-family="Instrument Serif, serif" '
        f'font-size="{font_size}" fill="#f3ebd7" '
        f'text-anchor="middle">{clean_display}</text>'
        f'<text x="100" y="260" font-family="JetBrains Mono, monospace" '
        f'font-size="7" fill="{p["accent"]}" text-anchor="middle">'
        f'{html.escape(str(director).upper()[:20])} // {year or ""}</text>'
        f'</svg>'
    )


def create_entry(title, theater, neighborhood, ticket_url, summary, format,
                 showtimes):
    clean_t = clean_film_title(title)
    tmdb_info = tmdb.fetch_movie(clean_t)

    display_title = tmdb_info["title"] if tmdb_info else clean_t
    director = tmdb_info["director"] if tmdb_info else "Unknown"
    year = tmdb_info["year"] if tmdb_info and tmdb_info.get("year") else "Classic"
    raw_summary = (tmdb_info["overview"] if tmdb_info
                   and tmdb_info.get("overview") else summary)
    clean_summary = trim_summary(raw_summary)

    return {
        "title": display_title,
        "director": director,
        "year": year,
        "theater": theater,
        "neighborhood": neighborhood,
        "matchScore": calculate_taste_score(display_title, director,
                                            clean_summary, tmdb_info),
        "seen": (display_title.lower() in watched_titles
                 or clean_t.lower() in watched_titles),
        "weekend": "current",
        "summary": clean_summary,
        "format": format,
        "ticketUrl": ticket_url,
        "showtimes": showtimes[:6],
        "poster": tmdb_info.get("poster") if tmdb_info else None,
        "svg": generate_poster_svg(display_title, director, year),
    }


# ---------------------------------------------------------------------------
# 5. Execute & write screenings.json
# ---------------------------------------------------------------------------
def main():
    if not SERPAPI_API_KEY:
        print("[Error] SERPAPI_API_KEY environment variable is missing. "
              "Check your workflow secrets.")
        sys.exit(1)

    final_dataset = fetch_serpapi_showtimes()

    if not final_dataset:
        print("[Engine Notice] 0 screenings retrieved. "
              "Verify API responses in the logs above.")
        sys.exit(1)

    # Sort: unwatched first, then by score descending, so the best
    # recommendations surface at the top.
    final_dataset.sort(key=lambda e: (e["seen"], -e["matchScore"]))

    payload = {
        "generated_at": datetime.datetime.now(
            datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "weekend": weekend_label,
        "screenings": final_dataset,
    }
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"[Engine] Wrote {len(final_dataset)} screenings to {OUTPUT_JSON}.")


if __name__ == "__main__":
    main()
