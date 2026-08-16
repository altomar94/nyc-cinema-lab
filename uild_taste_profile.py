import os
import csv
import json
import time
import urllib.parse
from collections import defaultdict
import requests

TMDB_API_KEY = os.environ.get("TMDB_API_KEY", "")
CSV_PATH = "ratings.csv"
OUTPUT_JSON = "taste_profile.json"

def get_rating_weight(stars):
    if stars >= 5.0: return 3.0
    if stars >= 4.5: return 2.0
    if stars >= 4.0: return 1.0
    if stars >= 3.5: return 0.5
    if stars >= 3.0: return -0.5
    if stars >= 2.5: return -1.0
    return -2.5

def fetch_tmdb_metadata(film_title, year=None):
    if not TMDB_API_KEY:
        return None
    try:
        query = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={urllib.parse.quote(film_title)}"
        if year and str(year).isdigit():
            query += f"&year={year}"
        
        res = requests.get(query, timeout=5).json()
        if not res.get('results'):
            return None
        
        movie = res['results'][0]
        movie_id = movie['id']
        
        # Crew & Credits
        credits_res = requests.get(f"https://api.themoviedb.org/3/movie/{movie_id}/credits?api_key={TMDB_API_KEY}", timeout=5).json()
        directors = [c['name'].lower() for c in credits_res.get('crew', []) if c.get('job') == 'Director']
        dps = [c['name'].lower() for c in credits_res.get('crew', []) if c.get('job') in ['Director of Photography', 'Cinematographer']]
        
        # Details & Keywords
        details_res = requests.get(f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}&append_to_response=keywords,reviews", timeout=5).json()
        overview = details_res.get('overview', '')
        keywords = [k['name'].lower() for k in details_res.get('keywords', {}).get('keywords', [])]
        reviews = [r['content'] for r in details_res.get('reviews', {}).get('results', [])[:2]]
        
        corpus = f"{overview} {' '.join(keywords)} {' '.join(reviews)}"
        return {'directors': directors, 'dps': dps, 'corpus': corpus}
    except Exception as e:
        print(f"Error fetching metadata for {film_title}: {e}")
        return None

def build_profile():
    if not os.path.exists(CSV_PATH):
        print(f"Error: Could not find {CSV_PATH}.")
        return

    watched_titles = []
    all_ratings = []
    director_affinity = defaultdict(float)
    dp_affinity = defaultdict(float)
    positive_corpus = []

    print(f"[1/3] Reading {CSV_PATH}...")
    with open(CSV_PATH, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"[2/3] Processing {len(rows)} films against TMDB...")
    for idx, row in enumerate(rows, 1):
        title = row.get('Name', '').strip()
        year = row.get('Year', '').strip()
        rating_raw = row.get('Rating', '').strip()
        
        if not title or not rating_raw:
            continue
        try:
            stars = float(rating_raw)
        except ValueError:
            continue
        
        watched_titles.append(title.lower())
        all_ratings.append(stars)
        weight = get_rating_weight(stars)

        # Query metadata for high/low rated films to build affinity weights
        if stars >= 4.0 or stars <= 2.5:
            meta = fetch_tmdb_metadata(title, year)
            if meta:
                for d in meta['directors']:
                    director_affinity[d] += weight
                for dp in meta['dps']:
                    dp_affinity[dp] += weight
                if weight > 0:
                    positive_corpus.append(meta['corpus'])
            time.sleep(0.05)  # Safe rate-limiting buffer

        if idx % 25 == 0 or idx == len(rows):
            print(f"  Processed {idx}/{len(rows)} films...")

    total_films = len(all_ratings)
    mean_rating = round(sum(all_ratings) / total_films, 2) if total_films else 3.79

    profile_data = {
        "total_films": total_films,
        "mean_rating": mean_rating,
        "watched_titles": list(set(watched_titles)),
        "director_affinity": dict(director_affinity),
        "dp_affinity": dict(dp_affinity),
        "positive_review_text": " ".join(positive_corpus) if positive_corpus else "nocturnal existential atmospheric crime neon-drenched stylized slow-burn"
    }

    print(f"[3/3] Writing compiled profile to {OUTPUT_JSON}...")
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(profile_data, f, indent=2)

    print(f"Done! {OUTPUT_JSON} successfully generated ({total_films} films, {mean_rating}★ avg).")

if __name__ == "__main__":
    build_profile()
