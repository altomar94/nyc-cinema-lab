import re
import os
import json
import zlib
import html
import datetime
import urllib.request
import urllib.parse
from collections import defaultdict
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# 1. Configuration & Target Weekend Dates
# ---------------------------------------------------------------------------
SERPAPI_API_KEY = os.environ.get("SERPAPI_API_KEY", "").strip()
TMDB_API_KEY = os.environ.get("TMDB_API_KEY", "").strip()
PROFILE_JSON = "taste_profile.json"

if not SERPAPI_API_KEY:
    print("[Error] SERPAPI_API_KEY environment variable is missing. Check your workflow secrets.")
    exit(1)

today = datetime.date.today()
days_until_friday = (4 - today.weekday()) % 7
if days_until_friday == 0 and today.weekday() != 4:
    days_until_friday = 7

friday_date = today + datetime.timedelta(days=days_until_friday)
saturday_date = friday_date + datetime.timedelta(days=1)
sunday_date = friday_date + datetime.timedelta(days=2)

fri_str = friday_date.strftime("%b %d")
weekend_range_label = f"{friday_date.strftime('%b %d')} – {sunday_date.strftime('%b %d')}"
print(f"[Calendar] Targeting weekend: {weekend_range_label}")

THEATER_MAP = {
    "amc lincoln square": ("AMC Lincoln Square 13", "Upper West Side", "https://www.amctheatres.com/movie-theatres/new-york-city/amc-lincoln-square-13"),
    "lincoln square 13": ("AMC Lincoln Square 13", "Upper West Side", "https://www.amctheatres.com/movie-theatres/new-york-city/amc-lincoln-square-13"),
    "lincoln square": ("AMC Lincoln Square 13", "Upper West Side", "https://www.amctheatres.com/movie-theatres/new-york-city/amc-lincoln-square-13"),
    "regal e-walk": ("Regal Times Square", "Times Square", "https://www.regmovies.com/theatres/regal-e-walk-times-square"),
    "regal times square": ("Regal Times Square", "Times Square", "https://www.regmovies.com/theatres/regal-e-walk-times-square"),
    "times square": ("Regal Times Square", "Times Square", "https://www.regmovies.com/theatres/regal-e-walk-times-square"),
    "film forum": ("Film Forum", "Greenwich Village", "https://filmforum.org/now_playing"),
    "ifc center": ("IFC Center", "West Village", "https://www.ifccenter.com/"),
    "metrograph": ("Metrograph", "Lower East Side", "https://metrograph.com/nyc/"),
    "paris theater": ("The Paris Theater", "Midtown", "https://www.paristheaternyc.com/"),
    "the paris": ("The Paris Theater", "Midtown", "https://www.paristheaternyc.com/"),
    "roxy cinema": ("The Roxy Cinema", "Tribeca", "https://www.roxycinematribeca.com/"),
    "film at lincoln center": ("Film at Lincoln Center", "Upper West Side", "https://www.filmlinc.org/now-playing/"),
    "walter reade": ("Film at Lincoln Center", "Upper West Side", "https://www.filmlinc.org/now-playing/"),
    "bam rose": ("BAM Rose Cinemas", "Brooklyn", "https://www.bam.org/film"),
    "bam": ("BAM Rose Cinemas", "Brooklyn", "https://www.bam.org/film"),
    "nitehawk": ("Nitehawk Cinema", "Brooklyn", "https://nitehawkcinema.com/"),
    "angelika": ("Angelika Film Center", "SoHo", "https://www.angelikafilmcenter.com/nyc"),
    "cinema village": ("Cinema Village", "Greenwich Village", "https://www.cinemavillage.com/")
}

STYLE_TROPES = [
    "nocturnal", "existential", "slow-burn", "kinetic", "neon", "melancholic",
    "paranoia", "isolation", "atmospheric", "stylized", "underworld", "obsession",
    "noir", "crime", "surreal", "laconic", "nihilistic", "poetic"
]

# ---------------------------------------------------------------------------
# 2. Taste Profile Loading
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

# ---------------------------------------------------------------------------
# 3. Clean Title & Smart Summary Trimming
# ---------------------------------------------------------------------------
tmdb_cache = {}

def clean_film_title(raw_title):
    t = raw_title.strip()
    t = re.sub(r'\(.*?\)|\[.*?\]', '', t)
    t = re.sub(r'\b(35mm|70mm|16mm|4k|restoration|restored|dcp|q&a|in person|repertory|special screening|preview|staff picks|with live score|waverly midnights|imax|rpx|3d)\b', '', t, flags=re.I)
    if " - " in t: t = t.split(" - ")[0]
    if " – " in t: t = t.split(" – ")[0]
    return re.sub(r'\s+', ' ', t).strip()

def trim_summary(text, max_chars=130):
    if not text:
        return ""
    text = text.strip()
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if sentences and 25 <= len(sentences[0]) <= max_chars:
        return sentences[0]
    if len(text) > max_chars:
        truncated = text[:max_chars].rsplit(' ', 1)[0]
        return truncated.rstrip('.,;:-') + '...'
    return text

def fetch_real_tmdb_metadata(film_title):
    clean_search = clean_film_title(film_title)
    clean_key = clean_search.lower()
    
    if clean_key in tmdb_cache:
        return tmdb_cache[clean_key]
        
    if not TMDB_API_KEY or len(clean_search) < 2:
        return None
    try:
        search_url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={urllib.parse.quote(clean_search)}"
        res = requests.get(search_url, timeout=6).json()
        results = res.get('results', [])
        if not results:
            tmdb_cache[clean_key] = None
            return None
        
        movie = results[0]
        movie_id = movie['id']
        release_date = movie.get('release_date', '')
        year = int(release_date.split('-')[0]) if (release_date and release_date.split('-')[0].isdigit()) else None
        poster_url = f"https://image.tmdb.org/t/p/w500{movie.get('poster_path')}" if movie.get('poster_path') else None
        
        credits_res = requests.get(f"https://api.themoviedb.org/3/movie/{movie_id}/credits?api_key={TMDB_API_KEY}", timeout=6).json()
        directors = [c['name'] for c in credits_res.get('crew', []) if c.get('job') == 'Director']
        dps = [c['name'] for c in credits_res.get('crew', []) if c.get('job') in ['Director of Photography', 'Cinematographer']]
        
        details_res = requests.get(f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}&append_to_response=keywords,reviews", timeout=6).json()
        overview = details_res.get('overview', '')
        keywords = [k['name'].lower() for k in details_res.get('keywords', {}).get('keywords', [])]
        reviews = [r['content'] for r in details_res.get('reviews', {}).get('results', [])[:2]]
        
        corpus = f"{overview} {' '.join(keywords)} {' '.join(reviews)}"
        data = {
            'title': movie.get('title', clean_search),
            'director': directors[0] if directors else 'Unknown',
            'directors': [d.lower() for d in directors],
            'dps': [dp.lower() for dp in dps],
            'year': year,
            'overview': overview,
            'corpus': corpus,
            'poster': poster_url
        }
        tmdb_cache[clean_key] = data
        return data
    except Exception:
        return None

# ---------------------------------------------------------------------------
# 4. Taste Scoring & Poster SVG Fallback
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
        dp_score = 0.0
        for dp in tmdb_info.get('dps', []):
            if dp in dp_affinity:
                dp_score += dp_affinity[dp] * 2.5
        score += max(-10.0, min(10.0, dp_score))
        
    screening_text = f"{summary} {tmdb_info['corpus'] if (tmdb_info and 'corpus' in tmdb_info) else ''}"
    try:
        if positive_review_text.strip():
            tfidf = TfidfVectorizer().fit_transform([positive_review_text, screening_text])
            sim = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
            score += round(sim * 14)
    except Exception:
        pass
        
    trope_count = sum(1 for trope in STYLE_TROPES if trope in screening_text.lower())
    score += min(round(trope_count * 2.5), 10)
    
    return max(30, min(int(score), 98))

def generate_poster_svg(title, director, year):
    h = zlib.crc32(title.encode('utf-8'))
    palettes = [
        {"bg": "#080507", "primary": "#ff2a4b", "secondary": "#e5a93c", "accent": "#00e5bc"},
        {"bg": "#04080e", "primary": "#00e5bc", "secondary": "#ff2a4b", "accent": "#f3ebd7"},
        {"bg": "#0c0608", "primary": "#e5a93c", "secondary": "#ff2a4b", "accent": "#8b93a6"},
        {"bg": "#06090c", "primary": "#ff2a4b", "secondary": "#00e5bc", "accent": "#e5a93c"}
    ]
    p = palettes[h % len(palettes)]
    clean_display = html.escape(title.upper())
    font_size = 11 if len(clean_display) > 22 else 14

    return (
        f'<svg viewBox="0 0 200 300" xmlns="http://www.w3.org/2000/svg">'
        f'<rect width="200" height="300" fill="{p["bg"]}"/>'
        f'<circle cx="100" cy="100" r="48" fill="{p["primary"]}" opacity="0.8"/>'
        f'<text x="100" y="238" font-family="Instrument Serif, serif" font-size="{font_size}" '
        f'fill="#f3ebd7" text-anchor="middle">{clean_display}</text>'
        f'<text x="100" y="260" font-family="JetBrains Mono, monospace" font-size="7" '
        f'fill="{p["accent"]}" text-anchor="middle">{html.escape(str(director).upper()[:20])} // {year or ""}</text>'
        f'</svg>'
    )

def create_entry(title, theater, neighborhood, ticket_url, summary, fmt, showtimes):
    clean_t = clean_film_title(title)
    tmdb_info = fetch_real_tmdb_metadata(clean_t)
    
    display_title = tmdb_info['title'] if (tmdb_info and tmdb_info.get('title')) else clean_t
    director = tmdb_info['director'] if (tmdb_info and tmdb_info.get('director')) else 'Unknown'
    year = tmdb_info['year'] if (tmdb_info and tmdb_info.get('year')) else 'Classic'
    raw_summary = tmdb_info['overview'] if (tmdb_info and tmdb_info.get('overview')) else summary
    clean_summary = trim_summary(raw_summary)
    
    poster = tmdb_info.get('poster') if tmdb_info else None
    match_score = calculate_taste_score(display_title, director, clean_summary, tmdb_info)
    
    return {
        "title": display_title,
        "director": director,
        "year": year,
        "theater": theater,
        "neighborhood": neighborhood,
        "matchScore": match_score,
        "seen": display_title.lower() in watched_titles or clean_t.lower() in watched_titles,
        "weekend": "current",
        "summary": clean_summary,
        "format": fmt,
        "ticketUrl": ticket_url,
        "showtimes": showtimes,
        "poster": poster,
        "svg": generate_poster_svg(display_title, director, year)
    }

# ---------------------------------------------------------------------------
# 5. Targeted SerpApi Ingestion Across All Tracked NYC Venues
# ---------------------------------------------------------------------------
def fetch_serpapi_showtimes():
    screenings_map = defaultdict(lambda: {
        'theater': None, 'neighborhood': None, 'ticket_url': None,
        'summary': '', 'format': 'DCP', 'showtimes': []
    })

    search_queries = [
        "AMC Lincoln Square 13 showtimes",
        "Regal Times Square showtimes",
        "IFC Center NYC showtimes",
        "Film Forum NYC showtimes",
        "Metrograph NYC showtimes",
        "Film at Lincoln Center showtimes",
        "The Paris Theater NYC showtimes",
        "Roxy Cinema Tribeca showtimes",
        "Angelika Film Center NYC showtimes",
        "Nitehawk Cinema Brooklyn showtimes",
        "BAM Rose Cinemas showtimes",
        "Cinema Village NYC showtimes"
    ]

    for q in search_queries:
        print(f"[SerpApi] Requesting showtimes for: '{q}'...")
        params = {
            "engine": "google",
            "q": q,
            "location": "New York, New York, United States",
            "hl": "en",
            "gl": "us",
            "api_key": SERPAPI_API_KEY
        }
        
        try:
            res = requests.get("https://serpapi.com/search.json", params=params, timeout=60)
            data = res.json()
            
            if "error" in data:
                print(f"[SerpApi API Error]: {data['error']}")
                continue

            showtimes_blocks = data.get("showtimes", [])
            if not showtimes_blocks and "knowledge_graph" in data:
                kg = data["knowledge_graph"]
                showtimes_blocks = kg.get("movies_results", []) or kg.get("theaters", [])

            if not showtimes_blocks:
                showtimes_blocks = data.get("movies_results", []) or data.get("local_results", [])

            for block in showtimes_blocks:
                raw_theater_name = block.get("name") or block.get("theater_name") or block.get("title", "")
                raw_theater_lower = raw_theater_name.lower()
                
                matched_venue = None
                for k, v in THEATER_MAP.items():
                    if k in raw_theater_lower or k in q.lower():
                        matched_venue = v
                        break
                        
                if not matched_venue:
                    continue

                t_name, neigh, t_url = matched_venue
                movies = block.get("movies", [])
                
                if not movies and ("showtimes" in block or "times" in block):
                    movies = [block]

                for m in movies:
                    raw_title = m.get("name") or m.get("title", "")
                    clean_t = clean_film_title(raw_title)
                    if len(clean_t) < 2 or clean_t.lower() in ["tickets", "directions", "website"]:
                        continue
                    
                    st_list = m.get("showtimes", []) or m.get("times", [])
                    time_strs = []
                    fmt = "Standard DCP"
                    
                    for st in st_list:
                        if isinstance(st, str):
                            time_strs.append(f"Fri {fri_str}: {st}")
                        elif isinstance(st, dict):
                            tm = st.get("time") or st.get("showtime")
                            if tm:
                                time_strs.append(f"Fri {fri_str}: {tm}")
                            fmt_type = st.get("type", "").lower()
                            if "70mm" in fmt_type or "imax" in fmt_type:
                                fmt = "70mm IMAX" if "70mm" in fmt_type else "IMAX Laser"
                            elif "35mm" in fmt_type:
                                fmt = "35mm Print"

                    if not time_strs:
                        time_strs = [f"Fri {fri_str}: Evening"]

                    key = (clean_t.lower(), t_name)
                    entry = screenings_map[key]
                    entry['title'] = clean_t
                    entry['theater'] = t_name
                    entry['neighborhood'] = neigh
                    entry['ticket_url'] = t_url
                    entry['format'] = fmt
                    entry['summary'] = f"Playing at {t_name}."
                    for ts in time_strs:
                        if ts not in entry['showtimes']:
                            entry['showtimes'].append(ts)

        except Exception as e:
            print(f"[SerpApi Error] Query '{q}' failed: {e}")

    results = []
    for (t_clean, theater_name), data in screenings_map.items():
        results.append(create_entry(
            title=data['title'],
            theater=data['theater'],
            neighborhood=data['neighborhood'],
            ticket_url=data['ticket_url'],
            summary=data['summary'],
            fmt=data['format'],
            showtimes=data['showtimes'][:4]
        ))

    print(f"[SerpApi Engine] Extracted {len(results)} verified live NYC screenings.")
    return results

# ---------------------------------------------------------------------------
# 6. Execute & Write to index.html
# ---------------------------------------------------------------------------
final_dataset = fetch_serpapi_showtimes()

if len(final_dataset) == 0:
    print("[Engine Notice] 0 screenings retrieved. Verify API response in logs.")
    exit(1)

with open("index.html", "r", encoding="utf-8") as f:
    html_content = f.read()

scraped_json = json.dumps(final_dataset, indent=4)
html_content = re.sub(
    r'const dataset = \[.*?\];',
    lambda _: f'const dataset = {scraped_json};',
    html_content,
    flags=re.DOTALL
)

with open("index.html", "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"[Engine] Successfully published {len(final_dataset)} screenings to index.html.")
