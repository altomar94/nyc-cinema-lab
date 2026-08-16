import re
import os
import json
import zlib
import datetime
import urllib.request
import urllib.parse
from collections import defaultdict
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# 1. Date & Configuration
# ---------------------------------------------------------------------------
TMDB_API_KEY = os.environ.get("TMDB_API_KEY", "")
PROFILE_JSON = "taste_profile.json"
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36'
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
    print(f"[Taste Engine] Loading cached profile from {PROFILE_JSON}...")
    with open(PROFILE_JSON, "r", encoding="utf-8") as f:
        profile = json.load(f)
    total_films = profile.get("total_films", 224)
    mean_rating = profile.get("mean_rating", 3.79)
    watched_titles = set(profile.get("watched_titles", []))
    director_affinity = profile.get("director_affinity", {})
    dp_affinity = profile.get("dp_affinity", {})
    positive_review_text = profile.get("positive_review_text", "")
else:
    print("[Taste Engine] taste_profile.json not found. Using baseline fallback.")
    total_films = 224
    mean_rating = 3.79
    watched_titles = set()
    director_affinity = {"wong kar-wai": 6.0, "jean-pierre melville": 4.0, "akira kurosawa": 3.0}
    dp_affinity = {"christopher doyle": 5.0}
    positive_review_text = "nocturnal existential atmospheric crime neon-drenched stylized slow-burn"

# ---------------------------------------------------------------------------
# 3. TMDB Helper for Screenings
# ---------------------------------------------------------------------------
tmdb_cache = {}

def fetch_screening_tmdb(film_title):
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
        
        first = res['results'][0]
        movie_id = first['id']
        poster_url = f"https://image.tmdb.org/t/p/w500{first.get('poster_path')}" if first.get('poster_path') else None
        
        credits_res = requests.get(f"https://api.themoviedb.org/3/movie/{movie_id}/credits?api_key={TMDB_API_KEY}", timeout=5).json()
        directors = [c['name'].lower() for c in credits_res.get('crew', []) if c.get('job') == 'Director']
        dps = [c['name'].lower() for c in credits_res.get('crew', []) if c.get('job') in ['Director of Photography', 'Cinematographer']]
        
        details_res = requests.get(f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}&append_to_response=keywords,reviews", timeout=5).json()
        overview = details_res.get('overview', '')
        keywords = [k['name'].lower() for k in details_res.get('keywords', {}).get('keywords', [])]
        reviews = [r['content'] for r in details_res.get('reviews', {}).get('results', [])[:2]]
        
        corpus = f"{overview} {' '.join(keywords)} {' '.join(reviews)}"
        data = {'directors': directors, 'dps': dps, 'corpus': corpus, 'poster': poster_url}
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
# 5. Poster SVGs & Screening Creation
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
    shape_type = h % 4
    
    if shape_type == 0:
        shape_svg = f'<circle cx="100" cy="100" r="65" fill="{p["primary"]}" opacity="0.85"/><circle cx="100" cy="100" r="40" stroke="{p["secondary"]}" stroke-width="2" fill="none"/>'
    elif shape_type == 1:
        shape_svg = f'<path d="M 0,220 L 140,40 L 170,40 L 30,220 Z" fill="{p["primary"]}" opacity="0.8"/><line x1="20" y1="20" x2="180" y2="240" stroke="{p["secondary"]}" stroke-width="1.5"/>'
    elif shape_type == 2:
        shape_svg = f'<rect x="35" y="45" width="130" height="130" stroke="{p["secondary"]}" stroke-width="1.5" fill="none"/><rect x="50" y="60" width="100" height="100" fill="{p["primary"]}" opacity="0.75"/>'
    else:
        shape_svg = f'<polygon points="100,35 165,110 100,185 35,110" fill="{p["primary"]}" opacity="0.8"/><circle cx="100" cy="110" r="25" fill="{p["secondary"]}"/>'

    return f'''<svg viewBox="0 0 200 300" xmlns="http://www.w3.org/2000/svg"><rect width="200" height="300" fill="{p["bg"]}"/>{shape_svg}<text x="100" y="240" font-family="Instrument Serif, serif" font-size="16" fill="#f3ebd7" text-anchor="middle" letter-spacing="1">{title.upper()[:20]}</text><text x="100" y="262" font-family="JetBrains Mono, monospace" font-size="7" fill="{p["accent"]}" text-anchor="middle" letter-spacing="1.5">{director.upper()[:22]} // {year}</text></svg>'''

def create_entry(title, director, year, theater, neighborhood, summary, fmt, showtimes, poster=None, ticket_url=None, weekend="current"):
    clean_t = title.strip()
    tmdb_info = fetch_screening_tmdb(clean_t)
    final_poster = tmdb_info.get('poster') if (tmdb_info and tmdb_info.get('poster')) else poster
    match_score = calculate_taste_score(clean_t, director, summary, tmdb_info)
    
    return {
        "title": clean_t,
        "director": director,
        "year": int(year) if str(year).isdigit() else 1980,
        "theater": theater,
        "neighborhood": neighborhood,
        "matchScore": match_score,
        "seen": clean_t.lower() in watched_titles,
        "weekend": weekend,
        "summary": summary,
        "format": fmt,
        "ticketUrl": ticket_url or "https://www.screenslate.com",
        "showtimes": showtimes,
        "poster": final_poster,
        "svg": generate_poster_svg(clean_t, director, year)
    }

FALLBACK_SCREENINGS = [
    create_entry("In the Mood for Love", "Wong Kar-wai", 2000, "Metrograph", "Lower East Side", "In 1962 Hong Kong, two neighbors form a delicate, unspoken bond after discovering their respective spouses are committing adultery.", "4K DCP", [f"Sat {sat_str}: 4:30 PM", f"Sun {sun_str}: 2:00 PM"], poster="https://m.media-amazon.com/images/M/MV5BYmVkNmIwYzgtMTk3Mi00MjhkLTk5NTgtNzA2Yjg0MDVjNzk1XkEyXkFqcGc@._V1_FMjpg_UX1000_.jpg", ticket_url="https://metrograph.com/nyc/"),
    create_entry("Fallen Angels", "Wong Kar-wai", 1995, "Metrograph", "Lower East Side", "The interconnected nocturnal lives of a weary hitman, his glamorous handler, and a mute eccentric collide across neon-drenched Hong Kong.", "35mm Print", [f"Fri {fri_str}: 10:00 PM", f"Sat {sat_str}: 9:30 PM", f"Sun {sun_str}: 7:15 PM"], poster="https://m.media-amazon.com/images/M/MV5BMDY4NTdhOGMtZmRiZC00MTY2LWI1MmYtMDNjYjRhNWZlMmIxXkEyXkFqcGc@._V1_FMjpg_UX1000_.jpg", ticket_url="https://metrograph.com/nyc/"),
    create_entry("Le Samourai", "Jean-Pierre Melville", 1967, "The Paris Theater", "Midtown", "A methodical Parisian hitman executes a contract with icy precision, setting off a ruthless police hunt and underworld betrayal.", "4K Restoration", [f"Fri {fri_str}: 8:00 PM", f"Sat {sat_str}: 6:00 PM", f"Sun {sun_str}: 3:30 PM"], poster="https://m.media-amazon.com/images/M/MV5BYWYwYWYyMDctZjFiNy00YmNmLWE1NmEtMmVkM2RlYzQ4Y2NhXkEyXkFqcGc@._V1_FMjpg_UX1000_.jpg", ticket_url="https://www.paristheaternyc.com/"),
    create_entry("Blow-Up", "Michelangelo Antonioni", 1966, "Film Forum", "Greenwich Village", "A mod London fashion photographer believes he has accidentally captured a murder in the background of a park photograph.", "35mm Print", [f"Fri {fri_str}: 6:30 PM", f"Sat {sat_str}: 8:20 PM", f"Sun {sun_str}: 4:10 PM"], poster="https://m.media-amazon.com/images/M/MV5BMjA4Nzg5NTY4N15BMl5BanBnXkFtZTcwNjc3ODgyMQ@@._V1_FMjpg_UX1000_.jpg", ticket_url="https://filmforum.org/now_playing"),
    create_entry("Throne of Blood", "Akira Kurosawa", 1957, "IFC Center", "West Village", "A warrior is driven to betrayal and bloody ambition by a prophetic spirit and his ruthless wife in feudal Japan.", "4K Restoration", [f"Fri {fri_str}: 7:00 PM", f"Sat {sat_str}: 4:15 PM", f"Sun {sun_str}: 6:30 PM"], poster="https://m.media-amazon.com/images/M/MV5BYjFjM2YyYjEtMjcwYi00NGQ2LWIzNGMtNTBhYTQ1YWRmNzNmXkEyXkFqcGc@._V1_FMjpg_UX1000_.jpg", ticket_url="https://www.ifccenter.com/"),
    create_entry("Oppenheimer", "Christopher Nolan", 2023, "AMC Lincoln Square 13", "Upper West Side", "A biographical drama detailing theoretical physicist J. Robert Oppenheimer and the Manhattan Project.", "70mm IMAX", [f"Fri {fri_str}: 6:45 PM", f"Sat {sat_str}: 2:30 PM", f"Sun {sun_str}: 7:15 PM"], poster="https://m.media-amazon.com/images/M/MV5BN2JkMDc5MGQtZjg3YS00NmFiLWIyZmQtZTJmNTM5MjVmYTQ4XkEyXkFqcGc@._V1_FMjpg_UX1000_.jpg", ticket_url="https://www.amctheatres.com/movie-theatres/new-york-city/amc-lincoln-square-13"),
    create_entry("Heat", "Michael Mann", 1995, "Regal Times Square", "Times Square", "A methodical thief and a relentless LAPD homicide detective engage in a lethal cat-and-mouse confrontation across Los Angeles.", "4K Laser RPX", [f"Sat {sat_str}: 8:30 PM", f"Sun {sun_str}: 5:15 PM"], poster="https://m.media-amazon.com/images/M/MV5BYjY1MDlhM2QtYmRkYS00Yjc5LWIwY2ItNmVkOWJjZDQ5MmU3XkEyXkFqcGc@._V1_FMjpg_UX1000_.jpg", ticket_url="https://www.regmovies.com/theatres/regal-e-walk-times-square")
]

# ---------------------------------------------------------------------------
# 6. Screen Slate Aggregator Scraper
# ---------------------------------------------------------------------------
def scrape_screen_slate():
    screenings = []
    url = "https://www.screenslate.com/listings"
    print(f"[Scraper] Querying Screen Slate listings from {url}...")
    
    try:
        res = requests.get(url, headers=HEADERS, timeout=12)
        if res.status_code != 200:
            print(f"[Scraper] Screen Slate returned status code {res.status_code}")
            return screenings
            
        soup = BeautifulSoup(res.text, 'lxml')
        
        # Parse screening blocks from Screen Slate
        articles = soup.select('article, .views-row, .listing, .daily-screening')
        for art in articles:
            # Title
            title_elem = art.select_one('h2, h3, .title, a[hreflang]')
            if not title_elem:
                continue
            title = title_elem.get_text(strip=True)
            if not title or len(title) < 2:
                continue
                
            # Venue Matching
            venue_text = art.get_text().lower()
            matched_theater = None
            neighborhood = "New York"
            ticket_url = "https://www.screenslate.com/listings"
            
            for key, (t_name, neigh, t_url) in THEATER_MAP.items():
                if key in venue_text:
                    matched_theater = t_name
                    neighborhood = neigh
                    ticket_url = t_url
                    break
                    
            if not matched_theater:
                continue
                
            # Summary / Description
            desc_elem = art.select_one('p, .field-body, .description')
            summary = desc_elem.get_text(strip=True) if desc_elem else "Special presentation / repertory exhibition in NYC."
            
            # Format parsing
            fmt = "35mm / DCP"
            if "70mm" in venue_text or "70mm" in summary.lower():
                fmt = "70mm Print"
            elif "35mm" in venue_text or "35mm" in summary.lower():
                fmt = "35mm Print"
            elif "imax" in venue_text or "imax" in summary.lower():
                fmt = "IMAX Laser"
            elif "4k" in venue_text or "4k" in summary.lower():
                fmt = "4K Restoration"
                
            # Showtimes
            showtimes = [f"Fri {fri_str}: 7:00 PM", f"Sat {sat_str}: 4:30 PM", f"Sun {sun_str}: 6:30 PM"]
            time_elems = art.select('.time, .showtime, time')
            if time_elems:
                showtimes = [t.get_text(strip=True) for t in time_elems[:3] if len(t.get_text(strip=True)) > 2]
                
            screenings.append(create_entry(
                title=title,
                director="Curated Selection",
                year=1985,
                theater=matched_theater,
                neighborhood=neighborhood,
                summary=summary[:160] + "..." if len(summary) > 160 else summary,
                fmt=fmt,
                showtimes=showtimes,
                ticket_url=ticket_url
            ))
            
    except Exception as e:
        print(f"[Scraper] Screen Slate error: {e}")
        
    print(f"[Scraper] Harvested {len(screenings)} listings from Screen Slate.")
    return screenings

# ---------------------------------------------------------------------------
# 7. Merge Screenings & Build Final Dataset
# ---------------------------------------------------------------------------
slate_screenings = scrape_screen_slate()

# Deduplicate by title + theater
seen_keys = set()
merged = []

for item in (slate_screenings + FALLBACK_SCREENINGS):
    key = f"{item['title'].lower()}_{item['theater'].lower()}"
    if key not in seen_keys:
        seen_keys.add(key)
        merged.append(item)

final_dataset = merged
print(f"[Engine] Total active screenings catalogued: {len(final_dataset)}")

# ---------------------------------------------------------------------------
# 8. Write to index.html
# ---------------------------------------------------------------------------
with open("index.html", "r", encoding="utf-8") as f:
    html = f.read()

# Update Top Bar Live Stats
html = re.sub(
    r'<div>\d+\s*FILMS LOGGED\s*//\s*(?:MEAN|AVERAGE)\s*RATING:\s*[\d\.]+\s*★</div>',
    f'<div>{total_films} FILMS LOGGED // AVERAGE RATING: {mean_rating} ★</div>',
    html
)

# Inject screening dataset
scraped_json = json.dumps(final_dataset, indent=4)
html = re.sub(r'const dataset = \[.*?\];', f'const dataset = {scraped_json};', html, flags=re.DOTALL)

with open("index.html", "w", encoding="utf-8") as f:
    f.write(html)

print(f"[Engine] Successfully updated index.html for {weekend_range_label}.")
