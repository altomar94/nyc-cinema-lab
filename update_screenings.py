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
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# 1. Target Dates (Upcoming Friday, Saturday, Sunday)
# ---------------------------------------------------------------------------
TMDB_API_KEY = os.environ.get("TMDB_API_KEY", "")
PROFILE_JSON = "taste_profile.json"
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
}

today = datetime.date.today()
days_until_friday = (4 - today.weekday()) % 7
if days_until_friday == 0 and today.weekday() != 4:
    days_until_friday = 7

friday_date = today + datetime.timedelta(days=days_until_friday)
saturday_date = friday_date + datetime.timedelta(days=1)
sunday_date = friday_date + datetime.timedelta(days=2)

target_dates = [friday_date, saturday_date, sunday_date]
date_labels = {
    friday_date.strftime("%Y-%m-%d"): f"Fri {friday_date.strftime('%b %d')}",
    saturday_date.strftime("%Y-%m-%d"): f"Sat {saturday_date.strftime('%b %d')}",
    sunday_date.strftime("%Y-%m-%d"): f"Sun {sunday_date.strftime('%b %d')}"
}

weekend_range_label = f"{friday_date.strftime('%b %d')} – {sunday_date.strftime('%b %d')}"
print(f"[Calendar] Scraping Screen Slate for: {weekend_range_label}")

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
    "museum of the moving image": ("Museum of the Moving Image", "Queens", "https://movingimage.org/"),
    "momi": ("Museum of the Moving Image", "Queens", "https://movingimage.org/"),
    "anthology": ("Anthology Film Archives", "East Village", "http://anthologyfilmarchives.org/"),
    "spectacle": ("Spectacle Theater", "Williamsburg", "https://www.spectacletheater.com/"),
    "nitehawk": ("Nitehawk Cinema", "Brooklyn", "https://nitehawkcinema.com/")
}

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
    total_films = 224
    mean_rating = 3.79
    watched_titles = set()
    director_affinity = {"wong kar-wai": 6.0, "jean-pierre melville": 4.0, "akira kurosawa": 3.0}
    dp_affinity = {"christopher doyle": 5.0}
    positive_review_text = "nocturnal existential atmospheric crime neon-drenched stylized slow-burn"

# ---------------------------------------------------------------------------
# 3. TMDB Metadata Fetcher
# ---------------------------------------------------------------------------
tmdb_cache = {}

def clean_film_title(raw_title):
    t = raw_title.strip()
    t = re.sub(r'\(.*?\)|\[.*?\]', '', t)
    t = re.sub(r'\b(35mm|70mm|16mm|4k|restoration|restored|dcp|q&a|in person|repertory|special screening|preview|staff picks|with live score)\b', '', t, flags=re.I)
    if " - " in t: t = t.split(" - ")[0]
    if " – " in t: t = t.split(" – ")[0]
    return re.sub(r'\s+', ' ', t).strip()

def fetch_real_tmdb_metadata(film_title, known_director=None, known_year=None):
    clean_search = clean_film_title(film_title)
    clean_key = clean_search.lower()
    
    if clean_key in tmdb_cache:
        return tmdb_cache[clean_key]
        
    if not TMDB_API_KEY or len(clean_search) < 2:
        return None
    try:
        search_url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={urllib.parse.quote(clean_search)}"
        if known_year and str(known_year).isdigit():
            search_url += f"&year={known_year}"
            
        res = requests.get(search_url, timeout=5).json()
        results = res.get('results', [])
        
        # Fallback search without year constraint if no direct hit
        if not results and known_year:
            search_url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={urllib.parse.quote(clean_search)}"
            res = requests.get(search_url, timeout=5).json()
            results = res.get('results', [])

        if not results:
            tmdb_cache[clean_key] = None
            return None
        
        movie = results[0]
        movie_id = movie['id']
        release_date = movie.get('release_date', '')
        year = int(release_date.split('-')[0]) if (release_date and release_date.split('-')[0].isdigit()) else known_year
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
            'director': directors[0] if directors else (known_director or 'Repertory Selection'),
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
        tfidf = TfidfVectorizer().fit_transform([positive_review_text, screening_text])
        sim = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
        score += round(sim * 14)
    except Exception:
        score += 4
        
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

def create_entry(title, director, year, theater, neighborhood, ticket_url, summary, fmt, showtimes):
    clean_t = clean_film_title(title)
    tmdb_info = fetch_real_tmdb_metadata(clean_t, known_director=director, known_year=year)
    
    display_title = tmdb_info['title'] if (tmdb_info and tmdb_info.get('title')) else clean_t
    final_director = tmdb_info['director'] if (tmdb_info and tmdb_info.get('director')) else (director or 'Repertory Selection')
    final_year = tmdb_info['year'] if (tmdb_info and tmdb_info.get('year')) else (year or 'Classic')
    final_summary = tmdb_info['overview'] if (tmdb_info and tmdb_info.get('overview')) else summary
    poster = tmdb_info.get('poster') if tmdb_info else None
    match_score = calculate_taste_score(display_title, final_director, final_summary, tmdb_info)
    
    return {
        "title": display_title,
        "director": final_director,
        "year": final_year,
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
        "svg": generate_poster_svg(display_title, final_director, final_year)
    }

# ---------------------------------------------------------------------------
# 5. Screen Slate Weekend Scraper
# ---------------------------------------------------------------------------
def scrape_screen_slate():
    """Scrapes Screen Slate daily calendar listings for target weekend dates."""
    screenings_map = defaultdict(lambda: {
        'director': None, 'year': None, 'theater': None,
        'neighborhood': None, 'ticket_url': None, 'summary': None,
        'format': 'DCP', 'showtimes': []
    })
    
    # Scrape each target date (e.g. 2026-08-21, 2026-08-22, 2026-08-23)
    urls_to_scrape = [
        ("https://www.screenslate.com/listings", "Today")
    ]
    for d in target_dates:
        d_str = d.strftime("%Y-%m-%d")
        urls_to_scrape.append((f"https://www.screenslate.com/listings/{d_str}", date_labels[d_str]))

    for url, day_label in urls_to_scrape:
        try:
            print(f"[Scraper] Querying {url}...")
            res = requests.get(url, headers=HEADERS, timeout=12)
            if res.status_code != 200:
                continue
                
            soup = BeautifulSoup(res.text, 'lxml')
            
            # Find venue sections or listing blocks
            venues = soup.select('.venue-section, .views-row, article, .daily-listing')
            if not venues:
                venues = [soup]

            current_venue_info = None

            for elem in soup.find_all(['h2', 'h3', 'h4', 'article', 'li', 'div']):
                text = elem.get_text(" ", strip=True)
                
                # Check for venue header
                for k, v in THEATER_MAP.items():
                    if k in text.lower() and len(text) < 60:
                        current_venue_info = v
                        break
                        
                if not current_venue_info:
                    continue

                # Match film title & details inside this venue section
                title_elem = elem.select_one('a[href*="/film/"], a[href*="/event/"], strong, .title')
                if not title_elem:
                    continue
                    
                raw_title = clean_film_title(title_elem.get_text(strip=True))
                if len(raw_title) < 2 or raw_title.lower() in ["read more", "buy tickets", "tickets", "membership"]:
                    continue
                
                # Format detection
                fmt = "DCP"
                if "70mm" in text.lower(): fmt = "70mm Print"
                elif "35mm" in text.lower(): fmt = "35mm Print"
                elif "16mm" in text.lower(): fmt = "16mm Print"
                elif "4k" in text.lower() or "restoration" in text.lower(): fmt = "4K Restoration"
                
                # Times extraction (e.g. 7:00pm, 9:30pm)
                time_matches = re.findall(r'\b\d{1,2}:\d{2}\s*(?:am|pm)\b', text, re.I)
                times = [f"{day_label}: {t.upper()}" for t in time_matches] if time_matches else [f"{day_label}: See Schedule"]
                
                # Year extraction if present
                year_match = re.search(r'\b(19\d\d|20[0-2]\d)\b', text)
                year = int(year_match.group(1)) if year_match else None
                
                # Store entry
                t_name, neigh, t_url = current_venue_info
                key = (raw_title.lower(), t_name)
                
                entry = screenings_map[key]
                entry['title'] = raw_title
                entry['theater'] = t_name
                entry['neighborhood'] = neigh
                entry['ticket_url'] = t_url
                entry['year'] = year or entry['year']
                entry['format'] = fmt
                entry['summary'] = f"Playing at {t_name}."
                for t in times:
                    if t not in entry['showtimes']:
                        entry['showtimes'].append(t)
                        
        except Exception as e:
            print(f"[Scraper] Error scraping {url}: {e}")

    results = []
    for (t_clean, theater_name), data in screenings_map.items():
        if data['showtimes']:
            results.append(create_entry(
                title=data['title'],
                director=data['director'],
                year=data['year'],
                theater=data['theater'],
                neighborhood=data['neighborhood'],
                ticket_url=data['ticket_url'],
                summary=data['summary'],
                fmt=data['format'],
                showtimes=data['showtimes'][:4]
            ))

    print(f"[Scraper] Screen Slate: {len(results)} verified NYC screenings gathered.")
    return results

# ---------------------------------------------------------------------------
# 6. Execute & Write to index.html
# ---------------------------------------------------------------------------
final_dataset = scrape_screen_slate()
print(f"[Engine] Total active screenings catalogued: {len(final_dataset)}")

with open("index.html", "r", encoding="utf-8") as f:
    html_content = f.read()

# Update Letterboxd Stats
html_content = re.sub(
    r'<div>\d+\s*FILMS LOGGED\s*//\s*(?:MEAN|AVERAGE)\s*RATING:\s*[\d\.]+\s*★</div>',
    lambda _: f'<div>{total_films} FILMS LOGGED // AVERAGE RATING: {mean_rating} ★</div>',
    html_content
)

# Inject verified dataset
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
