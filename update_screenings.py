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
# 1. Date & Configuration
# ---------------------------------------------------------------------------
TMDB_API_KEY = os.environ.get("TMDB_API_KEY", "")
PROFILE_JSON = "taste_profile.json"
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

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

THEATER_TICKET_URLS = {
    "Film Forum": "https://filmforum.org/now_playing",
    "IFC Center": "https://www.ifccenter.com/",
    "Metrograph": "https://metrograph.com/nyc/",
    "The Paris Theater": "https://www.paristheaternyc.com/",
    "The Roxy Cinema": "https://www.roxycinematribeca.com/",
    "Film at Lincoln Center": "https://www.filmlinc.org/now-playing/",
    "BAM Rose Cinemas": "https://www.bam.org/film"
}

# ---------------------------------------------------------------------------
# 2. Instant Taste Profile Loading (0 API Calls for Profile)
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
# 3. TMDB Helper for Active Screenings
# ---------------------------------------------------------------------------
def fetch_screening_tmdb(film_title):
    if not TMDB_API_KEY:
        return None
    try:
        search_url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={urllib.parse.quote(film_title)}"
        res = requests.get(search_url, timeout=5).json()
        if not res.get('results'):
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
        return {'directors': directors, 'dps': dps, 'corpus': corpus, 'poster': poster_url}
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
    final_ticket = ticket_url or THEATER_TICKET_URLS.get(theater, "#")
    
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
        "ticketUrl": final_ticket,
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
    create_entry("Lady Snowblood", "Toshiya Fujita", 1973, "The Roxy Cinema", "Tribeca", "A young woman raised from birth as an assassin seeks ruthless vengeance against the four criminals who destroyed her family in Meiji-era Japan.", "35mm Print", [f"Fri {fri_str}: 9:15 PM", f"Sat {sat_str}: 7:00 PM"], poster="https://m.media-amazon.com/images/M/MV5BNTI2ZmExNzQtYmFlMC00MDU2LTg0ZTUtZTFhNDBkYTAyZjljXkEyXkFqcGc@._V1_FMjpg_UX1000_.jpg", ticket_url="https://www.roxycinematribeca.com/"),
    create_entry("The Long Goodbye", "Robert Altman", 1973, "BAM Rose Cinemas", "Brooklyn", "PI Philip Marlowe mumbles his way through a hazy, sun-bleached 1970s Los Angeles while trying to clear a friend's name in a murder inquiry.", "35mm Print", [f"Sat {sat_str}: 6:30 PM", f"Sun {sun_str}: 4:00 PM"], poster="https://m.media-amazon.com/images/M/MV5BOTU2MTc3NWItM2I0Ny00OTc5LTgzNWItNzI5NmZkMjY4YjMzXkEyXkFqcGc@._V1_FMjpg_UX1000_.jpg", ticket_url="https://www.bam.org/film"),
    create_entry("Deep Red", "Dario Argento", 1975, "IFC Center", "West Village", "A jazz pianist and an inquisitive journalist investigate the grisly murder of a psychic medium in a baroque Italian town.", "Archival 35mm", [f"Fri {fri_str}: 11:45 PM", f"Sat {sat_str}: 11:45 PM"], poster="https://m.media-amazon.com/images/M/MV5BMTY3NTUxNTc4OV5BMl5BanBnXkFtZTcwMjI5Njg3OA@@._V1_FMjpg_UX1000_.jpg", ticket_url="https://www.ifccenter.com/"),
    create_entry("Branded to Kill", "Seijun Suzuki", 1967, "Metrograph", "Lower East Side", "A hitman with a fetish for sniffing boiling rice fails an assignment and becomes the target of a mysterious rival hitman.", "35mm Print", [f"Sat {sat_str}: 10:15 PM", f"Sun {sun_str}: 8:45 PM"], poster="https://m.media-amazon.com/images/M/MV5BYzA2NTcxMjAtMTY1NS00ZWI4LTk2MWUtZjEzYzA1MDRmNGVmXkEyXkFqcGc@._V1_FMjpg_UX1000_.jpg", ticket_url="https://metrograph.com/nyc/"),
    create_entry("Klute", "Alan J. Pakula", 1971, "The Paris Theater", "Midtown", "A small-town detective searches for a missing executive in New York City with the help of a high-class call girl who is being stalked.", "35mm Print", [f"Fri {fri_str}: 5:30 PM", f"Sun {sun_str}: 6:00 PM"], poster="https://m.media-amazon.com/images/M/MV5BMTUwMDYxMzk1MF5BMl5BanBnXkFtZTgwNTU5MzIyMDE@._V1_FMjpg_UX1000_.jpg", ticket_url="https://www.paristheaternyc.com/"),
    create_entry("Night on Earth", "Jim Jarmusch", 1991, "Film Forum", "Greenwich Village", "A collection of five vignettes unfolding simultaneously inside taxicabs across Los Angeles, New York, Paris, Rome, and Helsinki.", "35mm Print", [f"Fri {fri_str}: 9:00 PM", f"Sat {sat_str}: 9:00 PM"], poster="https://m.media-amazon.com/images/M/MV5BY2FmZTU4NzAtMGQxMy00OTQ1LWE2NTgtN2UyOGQwMzI3Y2ZiXkEyXkFqcGc@._V1_FMjpg_UX1000_.jpg", ticket_url="https://filmforum.org/now_playing")
]

# ---------------------------------------------------------------------------
# 6. Scrapers
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
                [f"Fri {fri_str}: 7:00 PM", f"Sat {sat_str}: 4:30 PM", f"Sun {sun_str}: 6:15 PM"],
                ticket_url="https://filmforum.org/now_playing"
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
                    [f"Fri {fri_str}: 8:15 PM", f"Sat {sat_str}: 5:00 PM", f"Sun {sun_str}: 7:30 PM"],
                    ticket_url="https://metrograph.com/nyc/"
                ))
    except Exception as e:
        print(f"[Scraper] Metrograph error: {e}")
    return results

scraped_list = []
scraped_list.extend(scrape_film_forum())
scraped_list.extend(scrape_metrograph())

final_dataset = scraped_list if len(scraped_list) > 0 else FALLBACK_SCREENINGS
print(f"[Engine] Total active screenings injected: {len(final_dataset)}")

# ---------------------------------------------------------------------------
# 7. Write to index.html
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

print(f"[Engine] Updated index.html ({total_films} logged, {mean_rating}★ avg, weekend {weekend_range_label}).")
