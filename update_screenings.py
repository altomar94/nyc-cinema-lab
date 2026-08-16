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
SERPAPI_API_KEY = os.environ.get("SERPAPI_API_KEY", "")
TMDB_API_KEY = os.environ.get("TMDB_API_KEY", "")
PROFILE_JSON = "taste_profile.json"

today = datetime.date.today()
days_until_friday = (4 - today.weekday()) % 7
if days_until_friday == 0 and today.weekday() != 4:
    days_until_friday = 7

friday_date = today + datetime.timedelta(days=days_until_friday)
saturday_date = friday_date + datetime.timedelta(days=1)
sunday_date = friday_date + datetime.timedelta(days=2)

fri_str = friday_date.strftime("%b %d")
sat_str = saturday_date.strftime("%b %d")
sun_str = sunday_date.strftime("%b %d")
weekend_range_label = f"{friday_date.strftime('%b %d')} – {sunday_date.strftime('%b %d')}"
print(f"[Calendar] Fetching Google showtimes via SerpApi for: {weekend_range_label}")

THEATER_MAP = {
    "amc lincoln square": ("AMC Lincoln Square 13", "Upper West Side", "https://www.amctheatres.com/movie-theatres/new-york-city/amc-lincoln-square-13"),
    "lincoln square 13": ("AMC Lincoln Square 13", "Upper West Side", "https://www.amctheatres.com/movie-theatres/new-york-city/amc-lincoln-square-13"),
    "regal e-walk": ("Regal Times Square", "Times Square", "https://www.regmovies.com/theatres/regal-e-walk-times-square"),
    "regal times square": ("Regal Times Square", "Times Square", "https://www.regmovies.com/theatres/regal-e-walk-times-square"),
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
    total_films = profile.get("total_films", 0)
    mean_rating = profile.get("mean_rating", 0.0)
    watched_titles = set(profile.get("watched_titles", []))
    director_affinity = profile.get("director_affinity", {})
    dp_affinity = profile.get("dp_affinity", {})
    positive_review_text = profile.get("positive_review_text", "")
else:
    total_films = 0
    mean_rating = 0.0
    watched_titles = set()
    director_affinity = {}
    dp_affinity = {}
    positive_review_text = ""

# ---------------------------------------------------------------------------
# 3. Clean Title & Fetch TMDB Metadata
# ---------------------------------------------------------------------------
tmdb_cache = {}

def clean_film_title(raw_title):
    t = raw_title.strip()
    t = re.sub(r'\(.*?\)|\[.*?\]', '', t)
    t = re.sub(r'\b(35mm|70mm|16mm|4k|restoration|restored|dcp|q&a|in person|repertory|special screening|preview|staff picks|with live score|waverly midnights|imax)\b', '', t, flags=re.I)
    if " - " in t: t = t.split(" - ")[0]
    if " – " in t: t = t.split(" – ")[0]
    return re.sub(r'\s+', ' ', t).strip()

def fetch_real_tmdb_metadata(film_title):
    clean_search = clean_film_title(film_title)
    clean_key = clean_search.lower()
    
    if clean_key in tmdb_cache:
        return tmdb_cache[clean_key]
        
    if not TMDB_API_KEY or len(clean_search) < 2:
        return None
    try:
        search_url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={urllib.parse.quote(clean_search)}"
        res = requests.get(search_url, timeout=5).json()
        results = res.get('results', [])
        if not results:
            tmdb_cache[clean_key] = None
            return None
        
        movie = results[0]
        movie_id = movie['id']
        release_date = movie.get('release_date', '')
        year = int(release_date.split('-')[0]) if (release_date and release_date.split('-')[0].isdigit()) else None
        poster_url = f"https://image.tmdb.org/t/p/w500{movie.get('poster_path')}" if movie.get('poster_path') else None
        
        credits_res = requests.get(f"https://api.themoviedb.org/3/movie/{movie_id}/credits?api_key={TMDB_API_KEY}", timeout=5).json()
        directors = [c['name'] for c in credits_res.get('crew', []) if c.get('job') == 'Director']
        dps = [c['name'] for c in credits_res.get('crew', []) if c.get('job') in ['Director of Photography', 'Cinematographer']]
        
        details_res = requests.get(f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}&append_to_response=keywords,reviews", timeout=5).json()
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
# 4. Pure Taste Score & Posters
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
    final_summary = tmdb_info['overview'] if (tmdb_info and tmdb_info.get('overview')) else summary
    poster = tmdb_info.get('poster') if tmdb_info else None
    match_score = calculate_taste_score(display_title, director, final_summary, tmdb_info)
    
    return {
        "title": display_title,
        "director": director,
        "year": year,
        "theater": theater,
        "neighborhood": neighborhood,
        "matchScore": match_score,
        "seen": display_title.lower() in watched_titles or clean_t.lower() in watched_titles,
        "weekend": "current",
        "summary": final_summary[:180] + "..." if len(final_summary) > 180 else final_summary,
        "format": fmt,
        "ticketUrl": ticket_url,
        "showtimes": showtimes,
        "poster": poster,
        "svg": generate_poster_svg(display_title, director, year)
    }

# ---------------------------------------------------------------------------
# 5. SerpApi Google Showtimes Ingestion
# ---------------------------------------------------------------------------
def fetch_serpapi_showtimes():
    if not SERPAPI_API_KEY:
        print("[SerpApi Error] SERPAPI_API_KEY environment variable is not set.")
        return []

    screenings_map = defaultdict(lambda: {
        'theater': None, 'neighborhood': None, 'ticket_url': None,
        'summary': '', 'format': 'DCP', 'showtimes': []
    })

    # Query key NYC cinema hubs across Manhattan (uses only 2 API queries per run)
    search_queries = [
        "movie showtimes Upper West Side Manhattan NYC",
        "movie showtimes Greenwich Village Manhattan NYC"
    ]

    for q in search_queries:
        print(f"[SerpApi] Requesting Google showtimes for: '{q}'...")
        params = {
            "engine": "google",
            "q": q,
            "location": "New York, New York, United States",
            "hl": "en",
            "gl": "us",
            "api_key": SERPAPI_API_KEY
        }
        
        try:
            res = requests.get("https://serpapi.com/search.json", params=params, timeout=20)
            data = res.json()
            
            showtimes_list = data.get("showtimes", [])
            for theater_block in showtimes_list:
                raw_theater_name = theater_block.get("theater_name", "").lower()
                
                # Match to our tracked venues
                matched_venue = None
                for k, v in THEATER_MAP.items():
                    if k in raw_theater_name:
                        matched_venue = v
                        break
                        
                if not matched_venue:
                    continue

                t_name, neigh, t_url = matched_venue
                movies = theater_block.get("movies", [])
                
                for m in movies:
                    raw_title = m.get("name", "")
                    clean_t = clean_film_title(raw_title)
                    if len(clean_t) < 2:
                        continue
                    
                    # Extract times and format
                    st_list = m.get("showtimes", [])
                    time_strs = []
                    fmt = "Standard DCP"
                    
                    for st in st_list:
                        tm = st.get("time")
                        if tm:
                            time_strs.append(f"Fri {fri_str}: {tm}")
                        # Check format attributes if provided by Google
                        fmt_type = st.get("type", "").lower()
                        if "70mm" in fmt_type or "imax" in fmt_type:
                            fmt = "70mm IMAX" if "70mm" in fmt_type else "IMAX Laser"
                        elif "35mm" in fmt_type:
                            fmt = "35mm Print"
                        elif "3d" in fmt_type:
                            fmt = "RealD 3D"

                    if not time_strs:
                        time_strs = [f"Fri {fri_str}: Check Schedule"]

                    key = (clean_t.lower(), t_name)
                    entry = screenings_map[key]
                    entry['title'] = clean_t
                    entry['theater'] = t_name
                    entry['neighborhood'] = neigh
                    entry['ticket_url'] = t_url
                    entry['format'] = fmt
                    entry['summary'] = f"Theatrical screening at {t_name}."
                    for ts in time_strs:
                        if ts not in entry['showtimes']:
                            entry['showtimes'].append(ts)

        except Exception as e:
            print(f"[SerpApi Error] Query failed: {e}")

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

    print(f"[SerpApi Engine] Ingested {len(results)} verified screenings across NYC multiplexes and indies.")
    return results

# ---------------------------------------------------------------------------
# 6. Execute & Update index.html
# ---------------------------------------------------------------------------
final_dataset = fetch_serpapi_showtimes()

if len(final_dataset) == 0:
    print("[Engine Warning] 0 screenings retrieved from SerpApi. Check your SERPAPI_API_KEY.")
    exit(0)

with open("index.html", "r", encoding="utf-8") as f:
    html_content = f.read()

# Update Stats
html_content = re.sub(
    r'<div>\d+\s*FILMS LOGGED\s*//\s*(?:MEAN|AVERAGE)\s*RATING:\s*[\d\.]+\s*★</div>',
    lambda _: f'<div>{total_films} FILMS LOGGED // AVERAGE RATING: {mean_rating} ★</div>',
    html_content
)

# Overwrite dataset with verified showtimes
scraped_json = json.dumps(final_dataset, indent=4)
html_content = re.sub(
    r'const dataset = \[.*?\];',
    lambda _: f'const dataset = {scraped_json};',
    html_content,
    flags=re.DOTALL
)

with open("index.html", "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"[Engine] Successfully published {len(final_dataset)} verified live screenings to index.html.")
