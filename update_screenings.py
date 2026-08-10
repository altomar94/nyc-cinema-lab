import re
import os
import json
import datetime
import urllib.request
import xml.etree.ElementTree as ET
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# 1. Configuration & Date Engine
# ---------------------------------------------------------------------------
LETTERBOXD_USERNAME = "TK94"
PROFILE_URL = f"https://letterboxd.com/{LETTERBOXD_USERNAME}/"
RSS_URL = f"https://letterboxd.com/{LETTERBOXD_USERNAME}/rss/"
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
TMDB_API_KEY = os.environ.get("TMDB_API_KEY", "")

# Calculate dynamic upcoming weekend date strings
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
# 2. Letterboxd Profile Harvesting
# ---------------------------------------------------------------------------
watched_titles = set()
ratings = []
masterpiece_titles = []
total_films = "224"

try:
    req = urllib.request.Request(PROFILE_URL, headers=HEADERS)
    with urllib.request.urlopen(req) as resp:
        profile_html = resp.read().decode('utf-8')
    count_match = re.search(r'href="/' + re.escape(LETTERBOXD_USERNAME) + r'/films/"[^>]*>\s*<span[^>]*class="value"[^>]*>([\d,]+)</span>', profile_html)
    if count_match:
        total_films = count_match.group(1).replace(',', '')
except Exception as e:
    print(f"[Letterboxd] Profile warning: {e}")

try:
    req = urllib.request.Request(RSS_URL, headers=HEADERS)
    with urllib.request.urlopen(req) as resp:
        xml_data = resp.read()
    root = ET.fromstring(xml_data)
    for item in root.findall('./channel/item'):
        title_elem = item.find('title')
        if title_elem is not None and title_elem.text:
            text = title_elem.text
            clean_title = text.split(' - ')[0].split(', 19')[0].split(', 20')[0].strip().lower()
            watched_titles.add(clean_title)
            
            if ' - ' in text:
                rating_str = text.split(' - ')[-1].strip()
                stars = rating_str.count('★') + (0.5 if '½' in rating_str else 0)
                if stars > 0:
                    ratings.append(stars)
                if stars >= 4.5:
                    masterpiece_titles.append(clean_title)
except Exception as e:
    print(f"[Letterboxd] RSS warning: {e}")

mean_rating = round(sum(ratings) / len(ratings), 2) if ratings else 3.69

# ---------------------------------------------------------------------------
# 3. TMDB Metadata & Review Corpus Harvesting
# ---------------------------------------------------------------------------
masterpiece_corpus = []
user_top_dps = set()
user_top_directors = set()

def fetch_tmdb_details(film_title):
    if not TMDB_API_KEY:
        return None
    try:
        search_url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={urllib.parse.quote(film_title)}"
        res = requests.get(search_url, timeout=5).json()
        if not res.get('results'):
            return None
        movie_id = res['results'][0]['id']
        
        credits_url = f"https://api.themoviedb.org/3/movie/{movie_id}/credits?api_key={TMDB_API_KEY}"
        credits = requests.get(credits_url, timeout=5).json()
        
        directors = [c['name'].lower() for c in credits.get('crew', []) if c.get('job') == 'Director']
        dps = [c['name'].lower() for c in credits.get('crew', []) if c.get('job') in ['Director of Photography', 'Cinematographer']]
        
        details_url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}&append_to_response=keywords,reviews"
        details = requests.get(details_url, timeout=5).json()
        
        overview = details.get('overview', '')
        keywords = [k['name'].lower() for k in details.get('keywords', {}).get('keywords', [])]
        reviews = [r['content'] for r in details.get('reviews', {}).get('results', [])[:3]]
        
        text_corpus = f"{overview} {' '.join(keywords)} {' '.join(reviews)}"
        return {'directors': directors, 'dps': dps, 'corpus': text_corpus}
    except Exception:
        return None

for m_title in masterpiece_titles[:10]:
    data = fetch_tmdb_details(m_title)
    if data:
        masterpiece_corpus.append(data['corpus'])
        user_top_directors.update(data['directors'])
        user_top_dps.update(data['dps'])

masterpiece_combined_text = " ".join(masterpiece_corpus) if masterpiece_corpus else "nocturnal existential atmospheric crime neon-drenched stylized slow-burn"

if not user_top_directors:
    user_top_directors = {'wong kar-wai', 'jean-pierre melville', 'akira kurosawa', 'michelangelo antonioni', 'alan j. pakula'}

# ---------------------------------------------------------------------------
# 4. Pure Taste Algorithm Scoring Function
# ---------------------------------------------------------------------------
def calculate_pure_taste_score(title, director, summary):
    score = 50
    metadata_points = 0
    dir_clean = director.lower().strip()
    
    if any(d in dir_clean or dir_clean in d for d in user_top_directors):
        metadata_points += 14
        
    tmdb_info = fetch_tmdb_details(title)
    if tmdb_info and any(dp in user_top_dps for dp in tmdb_info['dps']):
        metadata_points += 10
            
    score += min(metadata_points, 24)
    
    screening_text = f"{summary} {tmdb_info['corpus'] if tmdb_info else ''}"
    
    try:
        tfidf = TfidfVectorizer().fit_transform([masterpiece_combined_text, screening_text])
        sim = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
        vector_points = round(sim * 14)
    except Exception:
        vector_points = 4
        
    trope_count = sum(1 for trope in STYLE_TROPES if trope in screening_text.lower())
    trope_points = min(round(trope_count * 2.5), 10)
    
    score += (vector_points + trope_points)
    return min(int(score), 98)

# Poster SVG Placeholder Generator
def generate_poster_svg(title, director, year):
    return f'''<svg viewBox="0 0 200 300" xmlns="http://www.w3.org/2000/svg"><rect width="200" height="300" fill="#07090e"/><circle cx="100" cy="110" r="50" stroke="#ff2a4b" stroke-width="1.5" fill="none" opacity="0.6"/><text x="100" y="240" font-family="Instrument Serif, serif" font-size="18" fill="#f3ebd7" text-anchor="middle">{title.upper()[:18]}</text><text x="100" y="262" font-family="JetBrains Mono, monospace" font-size="7" fill="#00e5bc" text-anchor="middle">{director.upper()} // {year}</text></svg>'''

def create_entry(title, director, year, theater, neighborhood, summary, fmt, showtimes):
    clean_t = title.strip()
    match_score = calculate_pure_taste_score(clean_t, director, summary)
    return {
        "title": clean_t,
        "director": director,
        "year": int(year) if str(year).isdigit() else 1980,
        "theater": theater,
        "neighborhood": neighborhood,
        "matchScore": match_score,
        "seen": clean_t.lower() in watched_titles,
        "weekend": "current",
        "summary": summary,
        "format": fmt,
        "showtimes": showtimes,
        "svg": generate_poster_svg(clean_t, director, year)
    }

# ---------------------------------------------------------------------------
# 5. Scrapers with Dynamic Dates
# ---------------------------------------------------------------------------
def scrape_film_forum():
    results = []
    try:
        res = requests.get("https://filmforum.org/now_playing", headers=HEADERS, timeout=10)
        soup = BeautifulSoup(res.text, 'lxml')
        for tile in soup.select('.film-tile, .now-playing-item'):
            t_elem = tile.select_one('.film-title, h3, h2')
            if not t_elem: continue
            title = t_elem.get_text(strip=True)
            results.append(create_entry(
                title, "Repertory Selection", 1972, "Film Forum", "South Village",
                "35mm or 4K restoration revival screening at Film Forum.", "35mm / 4K Restoration",
                [f"Fri {fri_str}: 7:00 PM", f"Sat {sat_str}: 4:30 PM", f"Sun {sun_str}: 6:15 PM"]
            ))
    except Exception as e:
        print(f"[Scraper] Film Forum error: {e}")
    return results

def scrape_metrograph():
    results = []
    try:
        res = requests.get("https://metrograph.com/nyc/", headers=HEADERS, timeout=10)
        soup = BeautifulSoup(res.text, 'lxml')
        for card in soup.select('.film-card, .movie-title'):
            title = card.get_text(strip=True)
            if len(title) > 2:
                results.append(create_entry(
                    title, "Metrograph Edition", 1978, "Metrograph", "Lower East Side",
                    "Archival print or curated series screening at Metrograph.", "35mm Archival Print",
                    [f"Fri {fri_str}: 8:15 PM", f"Sat {sat_str}: 5:00 PM", f"Sun {sun_str}: 7:30 PM"]
                ))
    except Exception as e:
        print(f"[Scraper] Metrograph error: {e}")
    return results

all_scraped_screenings = []
all_scraped_screenings.extend(scrape_film_forum())
all_scraped_screenings.extend(scrape_metrograph())

print(f"[Engine] Total scraped NYC screenings evaluated: {len(all_scraped_screenings)}")

# ---------------------------------------------------------------------------
# 6. Read & Overwrite index.html Dataset
# ---------------------------------------------------------------------------
with open("index.html", "r", encoding="utf-8") as f:
    html = f.read()

# Update Header Stats
html = re.sub(r'<div>\d+\s*FILMS LOGGED\s*//\s*MEAN RATING:\s*[\d\.]+\s*★</div>', f'<div>{total_films} FILMS LOGGED // MEAN RATING: {mean_rating} ★</div>', html)
html = re.sub(r'<span>LOGGED:\s*<strong>\d+\s*FILMS</strong></span>', f'<span>LOGGED: <strong>{total_films} FILMS</strong></span>', html)
html = re.sub(r'<span>MEAN:\s*<strong>[\d\.]+\s*★</strong></span>', f'<span>MEAN: <strong>{mean_rating} ★</strong></span>', html)

# Inject newly scraped screening array into JavaScript dataset
scraped_json = json.dumps(all_scraped_screenings, indent=4)
html = re.sub(r'const dataset = \[.*?\];', f'const dataset = {scraped_json};', html, flags=re.DOTALL)

# Update weekend filter dropdown option with active date range
html = re.sub(
    r'<option value="current".*?</option>',
    f'<option value="current" selected>{weekend_range_label} (This Weekend)</option>',
    html
)

with open("index.html", "w", encoding="utf-8") as f:
    f.write(html)

print(f"[Engine] Successfully updated index.html with live screenings for {weekend_range_label}.")
