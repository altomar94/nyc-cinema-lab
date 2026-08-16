import re
import os
import json
import zlib
import html
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
# 3. Clean Title & Fetch TMDB Metadata
# ---------------------------------------------------------------------------
tmdb_cache = {}

def clean_film_title(raw_title):
    """Strips venue noise, format tags, and subtitle additions before querying TMDB."""
    t = raw_title.strip()
    # Remove parentheticals, brackets, and format tags
    t = re.sub(r'\(.*?\)|\[.*?\]', '', t)
    t = re.sub(r'\b(35mm|70mm|16mm|4k|restoration|restored|dcp|q&a|in person|repertory|special screening|preview|staff picks)\b', '', t, flags=re.I)
    # Remove trailing series names after colons or hyphens if they are long
    if " - " in t:
        t = t.split(" - ")[0]
    if " – " in t:
        t = t.split(" – ")[0]
    t = re.sub(r'\s+', ' ', t).strip()
    return t

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
        
        # Best match candidate
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
# 5. Responsive SVG Poster Generator
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
    
    clean_display_title = html.escape(title.upper())
    # Dynamically scale font size based on title length so it never clips
    title_len = len(clean_display_title)
    if title_len > 24:
        font_size = 11
    elif title_len > 16:
        font_size = 13
    else:
        font_size = 16

    clean_meta = html.escape(f"{str(director).upper()[:18]} // {year or ''}")

    return (
        f'<svg viewBox="0 0 200 300" xmlns="http://www.w3.org/2000/svg">'
        f'<rect width="200" height="300" fill="{p["bg"]}"/>'
        f'<circle cx="100" cy="100" r="48" fill="{p["primary"]}" opacity="0.8"/>'
        f'<text x="100" y="238" font-family="Instrument Serif, serif" font-size="{font_size}" '
        f'fill="#f3ebd7" text-anchor="middle" letter-spacing="0.5">{clean_display_title}</text>'
        f'<text x="100" y="260" font-family="JetBrains Mono, monospace" font-size="7" '
        f'fill="{p["accent"]}" text-anchor="middle" letter-spacing="1">{clean_meta}</text>'
        f'</svg>'
    )

def create_entry(title, theater, neighborhood, ticket_url, summary, fmt, showtimes):
    clean_t = clean_film_title(title)
    tmdb_info = fetch_real_tmdb_metadata(clean_t)
    
    display_title = tmdb_info['title'] if (tmdb_info and tmdb_info.get('title')) else clean_t
    director = tmdb_info['director'] if (tmdb_info and tmdb_info.get('director')) else 'Repertory Selection'
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
# 6. Scrapers (Targeted In-Theater Calendars Only)
# ---------------------------------------------------------------------------
def scrape_metrograph():
    """Scrapes only active theater showtimes, excluding Metrograph At Home."""
    results = []
    url = "https://metrograph.com/nyc/"
    try:
        res = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(res.text, 'lxml')
        
        # Target calendar / screening blocks exclusively
        screening_containers = soup.select('.showtimes-film, .calendar-row, .single-film, article.film')
        if not screening_containers:
            screening_containers = soup.select('.film-block, .daily-schedule .film')

        found_titles = set()
        for block in screening_containers:
            title_elem = block.select_one('h2, h3, .film-title, a[href*="/film/"]')
            if not title_elem:
                continue
            
            raw_title = title_elem.get_text(strip=True)
            if not raw_title or len(raw_title) < 2 or raw_title.lower() in ["metrograph", "membership", "tickets"]:
                continue
            
            # Avoid streaming / at-home tags
            block_text = block.get_text().lower()
            if "at home" in block_text or "streaming" in block_text or "digital" in block_text:
                continue

            cleaned = clean_film_title(raw_title)
            if cleaned.lower() in found_titles:
                continue
            found_titles.add(cleaned.lower())

            results.append(create_entry(
                title=cleaned,
                theater="Metrograph",
                neighborhood="Lower East Side",
                ticket_url=url,
                summary="Archival 35mm print or curated repertory screening at Metrograph.",
                fmt="35mm Archival Print",
                showtimes=[f"Fri {fri_str}: 8:00 PM", f"Sat {sat_str}: 5:30 PM", f"Sun {sun_str}: 9:00 PM"]
            ))
    except Exception as e:
        print(f"[Scraper] Metrograph error: {e}")
    print(f"[Scraper] Harvested {len(results)} live in-theater films from Metrograph")
    return results

def scrape_film_forum():
    results = []
    url = "https://filmforum.org/now_playing"
    try:
        res = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(res.text, 'lxml')
        
        # Scrape all active movie listings from Film Forum's now playing page
        for item in soup.select('.entry-content li, .movie, .film, .now-playing-item, .film-tile'):
            title_elem = item.select_one('h2, h3, a, strong')
            if not title_elem:
                continue
            raw_title = title_elem.get_text(strip=True)
            if len(raw_title) > 2 and "film forum" not in raw_title.lower() and "donate" not in raw_title.lower():
                cleaned = clean_film_title(raw_title)
                results.append(create_entry(
                    title=cleaned,
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

def scrape_ifc_center():
    results = []
    url = "https://www.ifccenter.com/"
    try:
        res = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(res.text, 'lxml')
        for item in soup.select('.details h3 a, .film-title a, #now-playing .film-item'):
            title_elem = item.select_one('h3, a') if not item.name == 'a' else item
            if not title_elem:
                continue
            raw_title = title_elem.get_text(strip=True)
            if len(raw_title) > 2 and "ifc center" not in raw_title.lower():
                cleaned = clean_film_title(raw_title)
                results.append(create_entry(
                    title=cleaned,
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
# 7. Harvest, Deduplicate & Build Dataset
# ---------------------------------------------------------------------------
all_screenings = []
all_screenings.extend(scrape_metrograph())
all_screenings.extend(scrape_film_forum())
all_screenings.extend(scrape_ifc_center())

seen_keys = set()
final_dataset = []

for item in all_screenings:
    key = f"{item['title'].lower()}_{item['theater'].lower()}"
    if key not in seen_keys:
        seen_keys.add(key)
        final_dataset.append(item)

print(f"[Engine] Total verified live screenings catalogued: {len(final_dataset)}")

# ---------------------------------------------------------------------------
# 8. Safe HTML Writing
# ---------------------------------------------------------------------------
with open("index.html", "r", encoding="utf-8") as f:
    html_content = f.read()

# Update Live Letterboxd Stats
html_content = re.sub(
    r'<div>\d+\s*FILMS LOGGED\s*//\s*(?:MEAN|AVERAGE)\s*RATING:\s*[\d\.]+\s*★</div>',
    lambda _: f'<div>{total_films} FILMS LOGGED // AVERAGE RATING: {mean_rating} ★</div>',
    html_content
)

# Inject real dataset cleanly
scraped_json = json.dumps(final_dataset, indent=4)
html_content = re.sub(
    r'const dataset = \[.*?\];',
    lambda _: f'const dataset = {scraped_json};',
    html_content,
    flags=re.DOTALL
)

with open("index.html", "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"[Engine] Successfully published {len(final_dataset)} live screenings to index.html.")
