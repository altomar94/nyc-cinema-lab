"""Build taste_profile.json from a Letterboxd ratings export.

Reads ratings.csv (Name, Year, Rating), enriches highly-loved and strongly-
disliked films with TMDB metadata, and writes director/DP affinity weights
plus a positive text corpus used by update_screenings.py to score listings.

Run manually when ratings change:
    TMDB_API_KEY=... python build_taste_profile.py
"""
import csv
import json
import os
import time
from collections import defaultdict

from tmdb_client import TMDBClient

CSV_PATH = "ratings.csv"
OUTPUT_JSON = "taste_profile.json"


def get_rating_weight(stars):
    if stars >= 5.0:
        return 3.0
    if stars >= 4.5:
        return 2.0
    if stars >= 4.0:
        return 1.0
    if stars >= 3.5:
        return 0.5
    if stars >= 3.0:
        return -0.5
    if stars >= 2.5:
        return -1.0
    return -2.5


def build_profile():
    if not os.path.exists(CSV_PATH):
        print(f"Error: Could not find {CSV_PATH}.")
        raise SystemExit(1)

    tmdb = TMDBClient(os.environ.get("TMDB_API_KEY", ""))
    if not tmdb.configured:
        print("[Warning] TMDB_API_KEY not set: profile will have no "
              "director/DP affinities.")

    watched_titles = []
    all_ratings = []
    director_affinity = defaultdict(float)
    dp_affinity = defaultdict(float)
    positive_corpus = []

    print(f"[1/3] Reading {CSV_PATH}...")
    with open(CSV_PATH, mode="r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print(f"[2/3] Processing {len(rows)} films against TMDB...")
    for idx, row in enumerate(rows, 1):
        title = (row.get("Name") or "").strip()
        year = (row.get("Year") or "").strip()
        rating_raw = (row.get("Rating") or "").strip()

        if not title or not rating_raw:
            continue
        try:
            stars = float(rating_raw)
        except ValueError:
            continue

        watched_titles.append(title.lower())
        all_ratings.append(stars)
        weight = get_rating_weight(stars)

        # Only spend API calls on films with a strong signal either way.
        if stars >= 4.0 or stars <= 2.5:
            meta = tmdb.fetch_movie(title, year)
            if meta:
                for d in meta["directors"]:
                    director_affinity[d] += weight
                for dp in meta["dps"]:
                    dp_affinity[dp] += weight
                if weight > 0:
                    positive_corpus.append(meta["corpus"])
            time.sleep(0.05)  # polite rate-limiting buffer

        if idx % 25 == 0 or idx == len(rows):
            print(f"  Processed {idx}/{len(rows)} films...")

    if not all_ratings:
        print("Error: no usable ratings found in CSV.")
        raise SystemExit(1)

    mean_rating = round(sum(all_ratings) / len(all_ratings), 2)

    profile_data = {
        "total_films": len(all_ratings),
        "mean_rating": mean_rating,
        "watched_titles": sorted(set(watched_titles)),
        "director_affinity": dict(director_affinity),
        "dp_affinity": dict(dp_affinity),
        # Fallback vibe keywords if the corpus came back empty.
        "positive_review_text": (" ".join(positive_corpus) if positive_corpus
                                 else "nocturnal existential atmospheric "
                                      "crime neon-drenched stylized slow-burn"),
    }

    print(f"[3/3] Writing compiled profile to {OUTPUT_JSON}...")
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(profile_data, f, indent=2)

    print(f"Done! {OUTPUT_JSON} generated "
          f"({len(all_ratings)} films, {mean_rating}* avg).")


if __name__ == "__main__":
    build_profile()
