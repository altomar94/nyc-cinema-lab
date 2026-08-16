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
# 1. Configuration & Dates
# ---------------------------------------------------------------------------
TMDB_API_KEY = os.environ.get("TMDB_API_KEY", "")
PROFILE_JSON = "taste_profile.json"
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36'
}

STYLE_TROPES = [
    "nocturnal", "existential", "slow-burn", "kinetic", "neon", "melancholic",
    "paranoia", "isolation", "atmospheric", "stylized", "underworld", "obsession",
    "noir", "crime", "surreal", "laconic", "nihilistic", "poetic"
]

THEATER_MAP = {
    "amc lincoln square": ("AMC Lincoln Square 13", "Upper West Side", "https://www.amctheatres.com/movie-theatres/new-york-city/amc-lincoln-square-13"),
    "lincoln square": ("AMC Lincoln Square 13", "Upper West Side", "https://www.amctheatres.com/movie-theatres/new-york-city/amc-lincoln-square-13"),
    "regal times square": ("Regal Times Square", "Times Square", "https://www.regmovies.com/theatres/regal-e-walk-times-square"),
    "e-walk": ("Regal Times Square", "Times Square", "https://www.regmovies.com/theatres/regal-e-walk-times-square"),
    "film forum": ("Film Forum", "Greenwich Village", "https://filmforum.org/now_playing"),
    "ifc center": ("IFC Center", "West Village", "https://www.ifccenter.com/"),
    "metrograph": ("Metrograph", "Lower East Side", "https://metrograph.com/nyc/"),
    "paris theater": ("The Paris Theater", "Midtown", "https://www.paristheaternyc.com/"),
    "roxy cinema": ("The Roxy Cinema", "Tribeca", "https://www.roxycinematribeca.com/"),
    "film at lincoln center": ("Film at Lincoln Center", "Upper West Side", "https://www.filmlinc.org/now-playing/"),
    "walter reade": ("Film at Lincoln Center", "Upper West Side", "https://www.filmlinc.org/now-playing/"),
    "bam": ("BAM Rose Cinemas", "Brooklyn", "https://www.bam.org/film"),
    "anthology": ("Anthology Film Archives", "East Village", "http://anthologyfilmarchives.org/")
}

# ---------------------------------------------------------------------------
# 2. Taste Profile Loading
# ---------------------------------------------------------------------------
if os.path.exists(PROFILE_JSON):
    print(f"[Taste Engine] Loading profile from {PROFILE_JSON}...")
    with open(PROFILE_JSON, "r", encoding="utf-8") as f:
        profile = json.load(f)
    total_films = profile.get("total_films", 0)
    mean_rating = profile.get("mean_rating", 0.0)
    watched_titles = set(profile.get("watched_titles", []))
    director_affinity = profile.get("director_affinity", {})
    dp_affinity = profile.get("dp_affinity", {})
    positive_review_text = profile.get("positive_review_text", "")
else:
    print("[Taste Engine] taste_profile.json not found. Run build_taste_profile.py first.")
    total_films = 0
    mean_rating = 0.0
    watched_titles = set()
    director_affinity = {}
    dp_affinity = {}
    positive_review_text = "nocturnal existential atmospheric crime neon-drenched stylized slow-burn"

# ---------------------------------------------------------------------------
# 3. TMDB Metadata Fetcher (Pulls Real Director, DP, Year, Poster)
# ---------------------------------------------------------------------------
tmdb_cache = {}

def fetch_real_tmdb_metadata(film_title):
    clean_key = film_title.strip().lower()
    if clean_key in tmdb_cache:
        return tmdb_cache[clean_key]
        
    if not TMDB_API_KEY:
        return None
    try:
        search_url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={urllib.parse.quote(film_title)}"
        res = requests.get(search_url, timeout=5).json()
        if not res.get('results'):
            tmdb_cache[clean_key] = None
            return None
        
        movie = res['results'][0]
        movie_id = movie['id']
        release_date = movie.get('release_date', '')
        year = int(release_date.split('-')[0]) if release_date else None
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
        tmdb_cache[clean_key] = data
        return data
    except Exception:
        return None

# ---------------------------------------------------------------------------
# 4. Pure Taste Match Index Calculation
# ---------------------------------------------------------------------------
def calculate_taste_score(title, director, summary, tmdb_info=None):
    score = 50.0
    dir_clean = director.lower().strip()
    
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
        {"bg": "#06090c", "primary": "#ff2a4b", "secondary": "#00e5bc", "accent": "#e5a93c"},
        {"bg": "#0a0a0d", "primary": "#f3ebd7", "secondary": "#ff2a4b", "accent": "#00e5bc"}
    ]
    p = palettes[h % len(palettes)]
    return f'''<svg viewBox="0 0 200 300" xmlns="http://www.w3.org/2000/svg"><rect width="200" height="300" fill="{p["bg"]}"/><circle cx="100" cy="100" r="50" fill="{p["primary"]}" opacity="0.8"/><text x="100" y="240" font-family="Instrument Serif, serif" font-size="16" fill="#f3ebd7" text-anchor="middle">{title.upper()[:20]}</text><text x="100" y="262" font-family="JetBrains Mono, monospace" font-size="7" fill="{p["accent"]}" text-anchor="middle">{str(director).upper()[:22]} // {year or ''}</text></svg>'''

def create_entry(title, theater, neighborhood, ticket_url, summary, fmt, showtimes):
    clean_t = title.strip()
    tmdb_info = fetch_real_tmdb_metadata(clean_t)
    
    director = tmdb_info['director'] if (tmdb_info and tmdb_info.get('director')) else 'Unknown'
    year = tmdb_info['year'] if (tmdb_info and tmdb_info.get('year')) else 'N/A'
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
# 6. Live Screen Slate Ingestion (Real Data Only)
# ---------------------------------------------------------------------------
def scrape_screen_slate():
    screenings = []
    url = "https://www.screenslate.com/listings"
    print(f"[Scraper] Scraping verified listings from {url}...")
    
    try:
        res = requests.get(url, headers=HEADERS, timeout=12)
        if res.status_code != 200:
            print(f"[Scraper] Screen Slate returned HTTP {res.status_code}")
            return screenings
            
        soup = BeautifulSoup(res.text, 'lxml')
        articles = soup.select('article, .views-row, .listing, .daily-screening')
        
        for art in articles:
            title_elem = art.select_one('h2, h3, .title, a[hreflang]')
            if not title_elem:
                continue
            title = title_elem.get_text(strip=True)
            if not title or len(title) < 2:
                continue
                
            art_text = art.get_text().lower()
            matched_theater = None
            neighborhood = "New York"
            ticket_url = "https://www.screenslate.com/listings"
            
            for key, (t_name, neigh, t_url) in THEATER_MAP.items():
                if key in art_text:
                    matched_theater = t_name
                    neighborhood = neigh
                    ticket_url = t_url
                    break
                    
            if not matched_theater:
                continue
                
            # Extract real description if available
            desc_elem = art.select_one('p, .field-body, .description')
            summary = desc_elem.get_text(strip=True) if desc_elem else "Active NYC repertory presentation."
            
            # Format parsing
            fmt = "Standard DCP"
            if "70mm" in art_text: fmt = "70mm Print"
            elif "35mm" in art_text: fmt = "35mm Print"
            elif "16mm" in art_text: fmt = "16mm Print"
            elif "imax" in art_text: fmt = "IMAX Laser"
            elif "4k" in art_text: fmt = "4K Restoration"
                
            # Extract real showtimes
            time_elems = art.select('.time, .showtime, time')
            showtimes = [t.get_text(strip=True) for t in time_elems if len(t.get_text(strip=True)) > 2]
            if not showtimes:
                showtimes = ["See Venue Schedule"]
                
            screenings.append(create_entry(
                title=title,
                theater=matched_theater,
                neighborhood=neighborhood,
                ticket_url=ticket_url,
                summary=summary,
                fmt=fmt,
                showtimes=showtimes
            ))
            
    except Exception as e:
        print(f"[Scraper] Error during scraping: {e}")
        
    return screenings

# ---------------------------------------------------------------------------
# 7. Deduplicate & Build Live Dataset
# ---------------------------------------------------------------------------
live_screenings = scrape_screen_slate()

seen_keys = set()
final_dataset = []

for item in live_screenings:
    key = f"{item['title'].lower()}_{item['theater'].lower()}"
    if key not in seen_keys:
        seen_keys.add(key)
        final_dataset.append(item)

print(f"[Engine] Total verified live screenings catalogued: {len(final_dataset)}")

# ---------------------------------------------------------------------------
# 8. Write Directly to index.html
# ---------------------------------------------------------------------------
with open("index.html", "r", encoding="utf-8") as f:
    html = f.read()

# Update Top Bar Stats
html = re.sub(
    r'<div>\d+\s*FILMS LOGGED\s*//\s*(?:MEAN|AVERAGE)\s*RATING:\s*[\d\.]+\s*★</div>',
    f'<div>{total_films} FILMS LOGGED // AVERAGE RATING: {mean_rating} ★</div>',
    html
)

# Inject real dataset
scraped_json = json.dumps(final_dataset, indent=4)
html = re.sub(r'const dataset = \[.*?\];', f'const dataset = {scraped_json};', html, flags=re.DOTALL)

with open("index.html", "w", encoding="utf-8") as f:
    f.write(html)

print("[Engine] Successfully published live-only screenings to index.html.")
