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

import numpy as np
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
# 2b. Normalized affinity lookups + taste corpora
#
# Names are normalized (lowercased, punctuation stripped) and matched
# exactly — the old substring test ("kubrick" in name) could match
# unrelated directors sharing a syllable.
# ---------------------------------------------------------------------------
FALLBACK_CORPUS = ("nocturnal existential atmospheric crime neon-drenched "
                   "stylized slow-burn")


def _norm_name(name):
    return re.sub(r"[^a-z0-9]+", " ", str(name).lower()).strip()


director_lookup = {_norm_name(d): w for d, w in director_affinity.items()
                   if _norm_name(d)}
dp_lookup = {_norm_name(d): w for d, w in dp_affinity.items()
             if _norm_name(d)}

if (positive_review_text.strip()
        and positive_review_text.strip() != FALLBACK_CORPUS):
    # Prefer per-film documents (real IDF statistics); fall back to the
    # legacy joined corpus for profiles built before per-film docs existed.
    positive_docs = (profile.get("positive_corpus_docs")
                     or [positive_review_text])
else:
    # No real corpus (TMDB was unavailable at build time): skip text
    # similarity and let the explicit style prior below carry the weight.
    positive_docs = []
negative_docs = profile.get("negative_corpus_docs") or []

# ---------------------------------------------------------------------------
# 3. SerpApi ingestion with defensive parsing
# ---------------------------------------------------------------------------
TIME_RE = re.compile(r"\b\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)\b")
SKIP_TITLES = {"tickets", "directions", "website", "showtimes"}

WEEKDAYS = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5,
            "sun": 6}
WEEKEND_DATES = {friday_date, saturday_date, sunday_date}


def resolve_day(day_label):
    """Map a SerpApi day label to a real date.

    Google returns day-grouped blocks labeled 'Today', 'Tomorrow', or a
    weekday name ('Fri', 'Saturday', ...). Returns a datetime.date, or None
    when the label is missing/unrecognized.
    """
    if not day_label:
        return None
    key = day_label.strip().lower()
    if key == "today":
        return today
    if key == "tomorrow":
        return today + datetime.timedelta(days=1)
    if key[:3] in WEEKDAYS:
        delta = (WEEKDAYS[key[:3]] - today.weekday()) % 7
        return today + datetime.timedelta(days=delta)
    return None


def _as_list(value):
    return value if isinstance(value, list) else []


def iter_movie_blocks(data):
    """Yield (raw_theater_name, day_label, movie_dict) from a SerpApi response.

    Google's live shape is day-grouped: data["showtimes"] is a list of
    {"day": "Today"|"Fri"|..., "movies": [{"name": ..., "showing": [...]}]}.
    Older/alternate shapes (knowledge-graph variants, movies/local results)
    are still handled. The day label rides along so showtimes can later be
    pinned to real dates.
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
        day = block.get("day")
        movies = _as_list(block.get("movies"))
        if not movies and any(k in block for k in ("showtimes", "times",
                                                   "showtime", "showing")):
            movies = [block]  # the block itself describes one movie
        for movie in movies:
            if isinstance(movie, dict):
                yield theater_raw, day, movie


def _format_rank(format_type):
    """Return (label, rank) for a SerpApi showing type string."""
    fl = (format_type or "").lower()
    if "70mm" in fl:
        return "70mm", 5
    if "imax" in fl:
        return "IMAX", 4
    if "35mm" in fl:
        return "35mm Print", 3
    if "dolby" in fl:
        return "Dolby Cinema", 2
    if fl.strip() and "standard" not in fl:
        return format_type.strip(), 1
    return "Standard DCP", 1


def extract_times(movie):
    """Return (times, format_label) from a SerpApi movie dict.

    times is a list of (day_label_or_None, time_string). Handles the live
    shape — {"showing": [{"time": ["11:30am", ...], "type": "Standard"}]} —
    plus string times, {"time","type"} dicts, and day-grouped dicts.
    """
    raw = (movie.get("showing") or movie.get("showtimes")
           or movie.get("times") or movie.get("showtime"))
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
    best_fmt, best_rank = "Standard DCP", 0
    for day, item in pairs:
        if isinstance(item, str):
            chunks, ftype, item_day = [item], "", None
        elif isinstance(item, dict):
            t = item.get("time") or item.get("showtime") or ""
            chunks = t if isinstance(t, list) else [t]
            ftype = item.get("type") or ""
            item_day = item.get("day")
        else:
            continue
        fmt_label, rank = _format_rank(ftype)
        if rank > best_rank:
            best_fmt, best_rank = fmt_label, rank
        for chunk in chunks:
            if not isinstance(chunk, str):
                continue
            m = TIME_RE.search(chunk)
            if m:
                times.append((day or item_day, m.group(0)))
    return times, best_fmt


def match_theater(raw_name, query):
    haystacks = [(raw_name or "").lower(), (query or "").lower()]
    for key, venue in THEATER_MAP.items():
        if any(key in h for h in haystacks):
            return venue
    return None


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
        for theater_raw, block_day, movie in iter_movie_blocks(data):
            venue = match_theater(theater_raw, q)
            if not venue:
                continue
            raw_title = movie.get("name") or movie.get("title", "")
            clean_t = clean_film_title(raw_title)
            if len(clean_t) < 2 or clean_t.lower() in SKIP_TITLES:
                continue

            times, fmt = extract_times(movie)
            n_movies += 1
            if not times:
                zero_time_movies += 1

            # Pin each showtime to a real date; keep only the target
            # Fri-Sun weekend. Entries with no weekend showtimes are
            # dropped instead of getting placeholder chips.
            dated = []
            for day_label, tstr in times:
                d = resolve_day(day_label or block_day)
                if d is not None and d in WEEKEND_DATES:
                    dated.append((d, tstr))
            n_times += len(dated)
            if not dated:
                continue

            labels = [f"{d.strftime('%a %b %d')}: {t}"
                      for d, t in dated[:6]]

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
#
# Score = 50 base + director affinity (exact name match, +/-14)
#               + DP affinity (exact name match, +/-10)
#               + text similarity to liked films minus similarity to
#                 disliked films (+/-14, single batch TF-IDF over all
#                 candidate texts so IDF means something)
#               + explicit style-trope prior (+10 max), clamped to 30..98.
# ---------------------------------------------------------------------------
def _affinity_component(names, lookup, per_point, cap):
    """Sum affinity weights for exactly-matched names, clamped."""
    total, seen = 0.0, set()
    for name in names or []:
        key = _norm_name(name)
        if key and key not in seen and key in lookup:
            seen.add(key)
            total += lookup[key] * per_point
    return max(-cap, min(cap, total))


def trope_component(text):
    """Hand-tuned style prior: +2.5 per matched trope keyword, capped."""
    count = sum(1 for trope in STYLE_TROPES if trope in text.lower())
    return min(round(count * 2.5), 10)


def compute_text_scores(screening_texts):
    """Batch text similarity for every screening at once.

    Fits one TF-IDF vectorizer over the liked-film docs, disliked-film
    docs, and all screening texts, then scores each screening by its
    cosine similarity to the liked centroid minus its similarity to the
    disliked centroid. Returns a list of score deltas parallel to
    screening_texts.
    """
    n = len(screening_texts)
    if n == 0 or not positive_docs:
        return [0.0] * n
    try:
        tfidf = TfidfVectorizer(
            stop_words="english", ngram_range=(1, 2),
            min_df=1, max_features=5000,
        ).fit_transform(positive_docs + negative_docs + screening_texts)
    except ValueError:
        return [0.0] * n  # empty vocabulary

    n_pos, n_neg = len(positive_docs), len(negative_docs)
    pos_centroid = np.asarray(tfidf[:n_pos].mean(axis=0))
    screen = tfidf[n_pos + n_neg:]
    pos_sim = cosine_similarity(screen, pos_centroid).ravel()
    if n_neg:
        neg_centroid = np.asarray(tfidf[n_pos:n_pos + n_neg].mean(axis=0))
        neg_sim = cosine_similarity(screen, neg_centroid).ravel()
    else:
        neg_sim = [0.0] * n
    return [round(float(p - q) * 14, 1) for p, q in zip(pos_sim, neg_sim)]


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

    directors = (tmdb_info.get("directors", []) if tmdb_info
                 else [director])
    dps = tmdb_info.get("dps", []) if tmdb_info else []
    screening_text = (f"{clean_summary} "
                      f"{tmdb_info.get('corpus', '') if tmdb_info else ''}")

    # Base score: affinities + style prior. The batch text-similarity
    # component is added in main() once all screenings are known.
    base_score = (50.0
                  + _affinity_component(directors, director_lookup, 3.5, 14.0)
                  + _affinity_component(dps, dp_lookup, 2.5, 10.0)
                  + trope_component(screening_text))

    return {
        "title": display_title,
        "director": director,
        "year": year,
        "theater": theater,
        "neighborhood": neighborhood,
        "matchScore": base_score,  # text component added in main()
        "_text": screening_text,   # popped before writing screenings.json
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

    # Batch text-similarity scoring: one TF-IDF fit over every screening
    # so IDF statistics are meaningful, then fold the deltas into the
    # base scores and clamp to the published 30..98 range.
    texts = [entry.pop("_text", "") for entry in final_dataset]
    text_deltas = compute_text_scores(texts)
    for entry, delta in zip(final_dataset, text_deltas):
        entry["matchScore"] = max(30, min(98, int(entry["matchScore"] + delta)))

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
