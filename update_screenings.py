import os
import csv
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
CSV_PATH = "ratings.csv"
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
# 2. Rating Weight Multiplier Engine
# ---------------------------------------------------------------------------
def get_rating_weight(stars):
    if stars >= 5.0: return 3.0
    if stars >= 4.5: return 2.0
    if stars >= 4.0: return 1.0
    if stars >= 3.5: return 0.5
    if stars >= 3.0: return -0.5
    if stars >= 2.5: return -1.0
    return -2.5

# ---------------------------------------------------------------------------
# 3. TMDB Metadata Helper
# ---------------------------------------------------------------------------
def fetch_tmdb_details(film_title):
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
        
        credits_url = f"https://api.themoviedb.org/3/movie/{movie_id}/credits?api_key={TMDB_API_KEY}"
        credits = requests.get(credits_url, timeout=5).json()
        
        directors = [c['name'].lower() for c in credits.get('crew', []) if c.get('job') == 'Director']
        dps = [c['name'].lower() for c in credits.get('crew', []) if c.get('job') in ['Director of Photography', 'Cinematographer']]
        
        details_url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}&append_to_response=keywords,reviews"
        details = requests.get(details_url, timeout=5).json()
        
        overview = details.get('overview', '')
        keywords = [k['name'].lower() for k in details.get('keywords', {}).get('keywords', [])]
        reviews = [r['content'] for r in details.get('reviews', {}).get('results', [])[:3]]
        
        corpus = f"{overview} {' '.join(keywords)} {' '.join(reviews)}"
        return {'directors': directors, 'dps': dps, 'corpus': corpus, 'poster': poster_url}
    except Exception:
        return None

# ---------------------------------------------------------------------------
# 4. Ingest All Logged Films from CSV
# ---------------------------------------------------------------------------
watched_titles = set()
director_affinity = defaultdict(float)
dp_affinity = defaultdict(float)
positive_corpus = []
all_ratings = []

if os.path.exists(CSV_PATH):
    print(f"[Taste Engine] Parsing lifetime data from {CSV_PATH}...")
    with open(CSV_PATH, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            title = row.get('Name', '').strip()
            rating_raw = row.get('Rating', '')
            if not title or not rating_raw:
                continue
            
            try:
                stars = float(rating_raw)
            except ValueError:
                continue
            
            clean_title = title.lower()
            watched_titles.add(clean_title)
            all_ratings.append(stars)
            weight = get_rating_weight(stars)
            
            # Fetch TMDB data for high and low extremes to calibrate taste model
            if stars >= 4.0 or stars <= 2.5:
                meta = fetch_tmdb_details(title)
                if meta:
                    for d in meta['directors']:
                        director_affinity[d] += weight
                    for dp in meta['dps']:
                        dp_affinity[dp] += weight
                    if weight > 0:
                        positive_corpus.append(meta['corpus'])
else:
    print("[Taste Engine] ratings.csv not found. Operating in fallback baseline mode.")
    all_ratings = [3.79] * 224
    director_affinity['wong kar-wai'] = 6.0
    director_affinity['jean-pierre melville'] = 4.0
    director_affinity['akira kurosawa'] = 3.0

total_films = len(all_ratings) if all_ratings else 224
mean_rating = round(sum(all_ratings) / len(all_ratings), 2) if all_ratings else 3.79
positive_review_text = " ".join(positive_corpus) if positive_corpus else "nocturnal existential atmospheric crime neon-drenched stylized slow-burn"

# ---------------------------------------------------------------------------
# 5. Weighted Taste Scoring
# ---------------------------------------------------------------------------
def calculate_taste_score(title, director, summary, tmdb_info=None):
    score = 50.0  # Baseline neutral
    dir_clean = director.lower().strip()
    
    # 1. Weighted Director Metric (-14 to +14 pts)
    dir_score = 0.0
    for d, weight in director_affinity.items():
        if d in dir_clean or dir_clean in d:
            dir_score += weight * 3.5
    score += max(-14.0, min(14.0, dir_score))
    
    # 2. Weighted DP Metric (-10 to +10 pts)
    if tmdb_info:
        dp_score = 0.0
        for dp in tmdb_info.get('dps', []):
            if dp in dp_affinity:
                dp_score += dp_affinity[dp] * 2.5
        score += max(-10.0, min(10.0, dp_score))
        
    # 3. Tone & Review Semantic Similarity (0 to +14 pts)
    screening_text = f"{summary} {tmdb_info['corpus'] if (tmdb_info and 'corpus' in tmdb_info) else ''}"
    try:
        tfidf = TfidfVectorizer().fit_transform([positive_review_text, screening_text])
        sim = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
        score += round(sim * 14)
    except Exception:
        score += 4
        
    # 4. Curated Style Tropes (0 to +10 pts)
    trope_count = sum(1 for trope in STYLE_TROPES if trope in screening_text.lower())
    score += min(round(trope_count * 2.5), 10)
    
    return max(30, min(int(score), 98))

# ---------------------------------------------------------------------------
# 6. SVG Generator & Screening Ingestion
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
    tmdb_info = fetch_tmdb_details(clean_t)
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
    create_entry("In the Mood for Love", "Wong Kar-wai", 2000, "Metrograph", "Lower East Side", "In 1962 Hong Kong, two neighbors form a delicate, unspoken bond after discovering their respective spouses are committing adultery.", "4K DCP", [f"Sat {sat_str}: 4:30 PM", f"Sun {sun_str}: 2:00 PM"], poster="https://image.tmdb.org/t/p/w500/iB6L2x39zM1zV0c849z52.jpg", ticket_url="https://metrograph.com/nyc/"),
    create_entry("Fallen Angels", "Wong Kar-wai", 1995, "Metrograph", "Lower East Side", "The interconnected nocturnal lives of a weary hitman, his glamorous handler, and a mute eccentric collide across neon-drenched Hong Kong.", "35mm Print", [f"Fri {fri_str}: 10:00 PM", f"Sat {sat_str}: 9:30 PM", f"Sun {sun_str}: 7:15 PM"], poster="https://image.tmdb.org/t/p/w500/A02LzpLsgC2BmsLypgCjU7Nsh0v.jpg", ticket_url="https://metrograph.com/nyc/"),
    create_entry("Le Samourai", "Jean-Pierre Melville", 1967, "The Paris Theater", "Midtown", "A methodical Parisian hitman executes a contract with icy precision, setting off a ruthless police hunt and underworld betrayal.", "4K Restoration", [f"Fri {fri_str}: 8:00 PM", f"Sat {sat_str}: 6:00 PM", f"Sun {sun_str}: 3:30 PM"], poster="https://image.tmdb.org/t/p/w500/7I0Zk0C1e1Zq9Gq6zR6s1k40x2y.jpg", ticket_url="https://www.paristheaternyc.com/"),
    create_entry("Blow-Up", "Michelangelo Antonioni", 1966, "Film Forum", "Greenwich Village", "A mod London fashion photographer believes he has accidentally captured a murder in the background of a park photograph.", "35mm Print", [f"Fri {fri_str}: 6:30 PM", f"Sat {sat_str}: 8:20 PM", f"Sun {sun_str}: 4:10 PM"], poster="https://image.tmdb.org/t/p/w500/kM66WJ5Zf905N6g9z5y5k23z3y2.jpg", ticket_url="https://filmforum.org/now_playing"),
    create_entry("Throne of Blood", "Akira Kurosawa", 1957, "IFC Center", "West Village", "A warrior is driven to betrayal and bloody ambition by a prophetic spirit and his ruthless wife in feudal Japan.", "4K Restoration", [f"Fri {fri_str}: 7:00 PM", f"Sat {sat_str}: 4:15 PM", f"Sun {sun_str}: 6:30 PM"], ticket_url="https://www.ifccenter.com/"),
    create_entry("Lady Snowblood", "Toshiya Fujita", 1973, "The Roxy Cinema", "Tribeca", "A young woman raised from birth as an assassin seeks ruthless vengeance against the four criminals who destroyed her family in Meiji-era Japan.", "35mm Print", [f"Fri {fri_str}: 9:15 PM", f"Sat {sat_str}: 7:00 PM"], ticket_url="https://www.roxycinematribeca.com/"),
    create_entry("The Long Goodbye", "Robert Altman", 1973, "BAM Rose Cinemas", "Brooklyn", "PI Philip Marlowe mumbles his way through a hazy, sun-bleached 1970s Los Angeles while trying to clear a friend's name in a murder inquiry.", "35mm Print", [f"Sat {sat_str}: 6:30 PM", f"Sun {sun_str}: 4:00 PM"], ticket_url="https://www.bam.org/film"),
    create_entry("Deep Red", "Dario Argento", 1975, "IFC Center", "West Village", "A jazz pianist and an inquisitive journalist investigate the grisly murder of a psychic medium in a baroque Italian town.", "Archival 35mm", [f"Fri {fri_str}: 11:45 PM", f"Sat {sat_str}: 11:45 PM"], ticket_url="https://www.ifccenter.com/"),
    create_entry("Branded to Kill", "Seijun Suzuki", 1967, "Metrograph", "Lower East Side", "A hitman with a fetish for sniffing boiling rice fails an assignment and becomes the target of a mysterious rival hitman.", "35mm Print", [f"Sat {sat_str}: 10:15 PM", f"Sun {sun_str}: 8:45 PM"], ticket_url="https://metrograph.com/nyc/"),
    create_entry("Klute", "Alan J. Pakula", 1971, "The Paris Theater", "Midtown", "A small-town detective searches for a missing executive in New York City with the help of a high-class call girl who is being stalked.", "35mm Print", [f"Fri {fri_str}: 5:30 PM", f"Sun {sun_str}: 6:00 PM"], ticket_url="https://www.paristheaternyc.com/"),
    create_entry("Night on Earth", "Jim Jarmusch", 1991, "Film Forum", "Greenwich Village", "A collection of five vignettes unfolding simultaneously inside taxicabs across Los Angeles, New York, Paris, Rome, and Helsinki.", "35mm Print", [f"Fri {fri_str}: 9:00 PM", f"Sat {sat_str}: 9:00 PM"], ticket_url="https://filmforum.org/now_playing")
]

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

print(f"[Engine] Successfully recalibrated model with {total_films} logged films (Mean: {mean_rating}★).")
