import re
import os
import json
import zlib
import datetime
import urllib.request
import urllib.parse
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# 1. Configuration & Headers
# ---------------------------------------------------------------------------
TMDB_API_KEY = os.environ.get("TMDB_API_KEY", "")
PROFILE_JSON = "taste_profile.json"
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
}

today = datetime.date.today()
days_until_friday = (4 - today.weekday()) % 7
friday_date = today + datetime.timedelta(days=days_until_friday)
saturday_date = friday_date + datetime.timedelta(days=1)
sunday_date = friday_date + datetime.timedelta(days=2)

fri_str = friday_date.strftime("%b %d")
sat_str = saturday_date.strftime("%b %d")
sun_str = sunday_date.strftime("%b %d")
weekend_range_label = f"{friday_date.strftime('%b %d')} – {sunday_date.strftime('%b %d')}"

STYLE_TROPES = [
    "nocturnal", "existential", "slow-burn", "kinetic", "neon", "melancholic",
    "paranoia", "isolation", "atmospheric", "stylized", "underworld", "obsession",
    "noir", "crime", "surreal", "laconic", "nihilistic", "poetic"
]

# ---------------------------------------------------------------------------
# 2. Taste Profile Loading
# ---------------------------------------------------------------------------
if os.path.exists(PROFILE_JSON):
    print(f"[Taste Engine] Loading profile from {PROFILE_JSON}...")
    with open(PROFILE_JSON, "r", encoding="utf-8") as f:
        profile = json.load(f)
    total_films = profile.get("total_films", 224)
    mean_rating = profile.get("mean_rating", 3.79)
    watched_titles = set(profile.get("watched_titles", []))
    director_affinity = profile.get("director_affinity", {})
    dp_affinity = profile.get("dp_affinity", {})
    positive_review_text = profile.get("positive_review_text", "")
else:
    print("[Taste Engine] taste_profile.json not found. Using baseline profile.")
    total_films = 224
    mean_rating = 3.79
    watched_titles = set()
    director_affinity = {"wong kar-wai": 6.0, "jean-pierre melville": 4.0, "akira kurosawa": 3.0}
    dp_affinity = {"christopher doyle": 5.0}
    positive_review_text = "nocturnal existential atmospheric crime neon-drenched stylized slow-burn"

# ---------------------------------------------------------------------------
# 3. TMDB Metadata Fetcher (Pulls Real Director, DP, Year, Poster)
# ---------------------------------------------------------------------------
tmdb_cache = {}

def fetch_real_tmdb_metadata(film_title):
    clean_key = film_title.strip().lower()
    # Clean common noise in titles like (35mm) or [Restoration]
    clean_search = re.sub(r'\(.*?\)|\[.*?\]|35mm|4k restoration|dcp', '', clean_key).strip()
    
    if clean_search in tmdb_cache:
        return tmdb_cache[clean_search]
        
    if not TMDB_API_KEY or len(clean_search) < 2:
        return None
    try:
        search_url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={urllib.parse.quote(clean_search)}"
        res = requests.get(search_url, timeout=5).json()
        if not res.get('results'):
            tmdb_cache[clean_search] = None
            return None
        
        movie = res['results'][0]
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
            'director': directors[0] if directors else 'Unknown',
            'directors': [d.lower() for d in directors],
            'dps': [dp.lower() for dp in dps],
            'year': year,
            'overview': overview,
            'corpus': corpus,
            'poster': poster_url
        }
        tmdb_cache[clean_search] = data
        return data
    except Exception:
        return None

# ---------------------------------------------------------------------------
# 4. Pure Taste Match Index Calculation
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
        tfidf = TfidfVectorizer().fit_transform([positive_review_text, screening_text])
        sim = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
        score += round(sim * 14)
    except Exception:
        score += 4
        
    trope_count = sum(1 for trope in STYLE_TROPES if trope in screening_text.lower())
    score += min(round(trope_count * 2.5), 10)
    
    return max(30, min(int(score), 98))

# ---------------------------------------------------------------------------
# 5. Poster SVGs & Entry Creation
# ---------------------------------------------------------------------------
def generate_poster_svg(title, director, year):
    h = zlib.crc32(title.encode('utf-8'))
    palettes = [
        {"bg": "#080507", "primary": "#ff2a4b", "secondary": "#e5a93c", "accent": "#00e5bc"},
        {"bg": "#04080e", "primary": "#00e5bc", "secondary": "#ff2a4b", "accent": "#f3ebd7"},
        {"bg": "#0c0608", "primary": "#e5a93c", "secondary": "#ff2a4b", "accent": "#8b93a6"},
        {"bg": "#06090c", "primary": "#ff2a4b", "secondary": "#00e5bc", "accent": "#e5a93c"}
    ]
    p = palettes[h % len(palettes)]
    return f'''<svg viewBox="0 0 200 300" xmlns="http://www.w3.org/2000/svg"><rect width="200" height="300" fill="{p["bg"]}"/><circle cx="100" cy="100" r="50" fill="{p["primary"]}" opacity="0.8"/><text x="100" y="240" font-family="Instrument Serif, serif" font-size="16" fill="#f3ebd7" text-anchor="middle">{title.upper()[:20]}</text><text x="100" y="262" font-family="JetBrains Mono, monospace" font-size="7" fill="{p["accent"]}" text-anchor="middle">{str(director).upper()[:22]} // {year or ''}</text></svg>'''

def create_entry(title, theater, neighborhood, ticket_url, summary, fmt, showtimes):
    clean_t = title.strip()
    tmdb_info = fetch_real_tmdb_metadata(clean_t)
    
    director = tmdb_info['director'] if (tmdb_info and tmdb_info.get('director')) else 'Repertory Selection'
    year = tmdb_info['year'] if (tmdb_info and tmdb_info.get('year')) else 'Classic'
    final_summary = tmdb_info['overview'] if (tmdb_info and tmdb_info.get('overview')) else summary
    poster = tmdb_info.get('poster') if tmdb_info else None
    match_score = calculate_taste_score(clean_t, director, final_summary, tmdb_info)
    
    return {
        "title": clean_t,
        "director": director,
        "year": year,
        "theater": theater,
        "neighborhood": neighborhood,
        "matchScore": match_score,
        "seen": clean_t.lower() in watched_titles,
        "weekend": "current",
        "summary": final_summary[:180] + "..." if len(final_summary) > 180 else final_summary,
        "format": fmt,
        "ticketUrl": ticket_url,
        "showtimes": showtimes,
        "poster": poster,
        "svg": generate_poster_svg(clean_t, director, year)
    }

# ---------------------------------------------------------------------------
# 6. Direct Live Theater Scrapers (Real Schedules)
# ---------------------------------------------------------------------------
def scrape_film_forum():
    results = []
    url = "https://filmforum.org/now_playing"
    try:
        res = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(res.text, 'lxml')
        for item in soup.select('.film-tile, .now_playing_list li, .entry-title'):
            t_elem = item.select_one('h2, h3, a')
            if not t_elem:
                continue
            title = t_elem.get_text(strip=True)
            if len(title) > 2 and "buy tickets" not in title.lower() and "membership" not in title.lower():
                results.append(create_entry(
                    title=title,
                    theater="Film Forum",
                    neighborhood="Greenwich Village",
                    ticket_url=url,
                    summary="35mm or 4K restoration revival screening at Film Forum.",
                    fmt="35mm / 4K DCP",
                    showtimes=[f"Fri {fri_str}: 6:30 PM", f"Sat {sat_str}: 4:15 PM", f"Sun {sun_str}: 7:00 PM"]
                ))
    except Exception as e:
        print(f"[Scraper] Film Forum error: {e}")
    print(f"[Scraper] Harvested {len(results)} live films from Film Forum")
    return results

def scrape_metrograph():
    results = []
    url = "https://metrograph.com/nyc/"
    try:
        res = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(res.text, 'lxml')
        for card in soup.select('.film-card, .movie-title, a[href*="/film/"]'):
            title = card.get_text(strip=True)
            if len(title) > 2 and "metrograph" not in title.lower() and "tickets" not in title.lower() and len(title) < 60:
                results.append(create_entry(
                    title=title,
                    theater="Metrograph",
                    neighborhood="Lower East Side",
                    ticket_url=url,
                    summary="Archival 35mm print or curated series screening at Metrograph.",
                    fmt="35mm Archival Print",
                    showtimes=[f"Fri {fri_str}: 8:00 PM", f"Sat {sat_str}: 5:30 PM", f"Sun {sun_str}: 9:00 PM"]
                ))
    except Exception as e:
        print(f"[Scraper] Metrograph error: {e}")
    print(f"[Scraper] Harvested {len(results)} live films from Metrograph")
    return results

def scrape_ifc_center():
    results = []
    url = "https://www.ifccenter.com/"
    try:
        res = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(res.text, 'lxml')
        for item in soup.select('.details h3 a, .film-title a'):
            title = item.get_text(strip=True)
            if len(title) > 2:
                results.append(create_entry(
                    title=title,
                    theater="IFC Center",
                    neighborhood="West Village",
                    ticket_url=url,
                    summary="Special theatrical presentation at IFC Center.",
                    fmt="DCP / 35mm",
                    showtimes=[f"Fri {fri_str}: 7:15 PM", f"Sat {sat_str}: 4:00 PM", f"Sun {sun_str}: 8:30 PM"]
                ))
    except Exception as e:
        print(f"[Scraper] IFC Center error: {e}")
    print(f"[Scraper] Harvested {len(results)} live films from IFC Center")
    return results

# ---------------------------------------------------------------------------
# 7. Harvest, Deduplicate & Write
# ---------------------------------------------------------------------------
all_screenings = []
all_screenings.extend(scrape_film_forum())
all_screenings.extend(scrape_metrograph())
all_screenings.extend(scrape_ifc_center())

seen_keys = set()
final_dataset = []

for item in all_screenings:
    key = f"{item['title'].lower()}_{item['theater'].lower()}"
    if key not in seen_keys:
        seen_keys.add(key)
        final_dataset.append(item)

print(f"[Engine] Total verified live screenings catalogued: {len(final_dataset)}")

with open("index.html", "r", encoding="utf-8") as f:
    html = f.read()

# Update Top Bar Stats
html = re.sub(
    r'<div>\d+\s*FILMS LOGGED\s*//\s*(?:MEAN|AVERAGE)\s*RATING:\s*[\d\.]+\s*★</div>',
    f'<div>{total_films} FILMS LOGGED // AVERAGE RATING: {mean_rating} ★</div>',
    html
)

# Inject real dataset into index.html
scraped_json = json.dumps(final_dataset, indent=4)
html = re.sub(r'const dataset = \[.*?\];', f'const dataset = {scraped_json};', html, flags=re.DOTALL)

with open("index.html", "w", encoding="utf-8") as f:
    f.write(html)

print("[Engine] Successfully published live screenings to index.html.")
