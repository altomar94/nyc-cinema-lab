# nyc-cinema-lab

A personal site that lists upcoming NYC repertory / art-house screenings,
ranked by how well they match your taste. Built from a Letterboxd ratings
export + TMDB metadata. Live at **nycfilmscreenings.com** (GitHub Pages).

## How it works

1. **`build_taste_profile.py`** (run manually when your ratings change) reads
   `ratings.csv`, enriches strongly-loved/disliked films via TMDB, and writes
   `taste_profile.json`: director/DP affinity weights plus a text corpus of
   what you rate highly.
2. **`update_screenings.py`** (runs every Friday morning via GitHub Actions)
   scrapes Google showtimes for 11 tracked NYC venues through SerpApi,
   enriches each film with TMDB, scores it against your taste profile, and
   writes **`screenings.json`**. Scoring is used for ranking only — no
   percentages are shown:
   - **Director affinity** (±14): Letterboxd-rating-weighted, exact
     normalized name match — no substring false positives.
   - **Cinematographer affinity** (±10): same idea, smaller weight.
   - **Text similarity** (±14): one batch TF-IDF fit over all candidate
     listings; cosine to the liked-film centroid *minus* cosine to the
     disliked-film centroid, so films you hated actively push scores down.
   - **Style prior** (+10 max): hand-picked trope keywords
     ("noir", "slow-burn", …).
   - Listings are deduped to unique films (all venues/showtimes merged),
     ranked unwatched-first by score, and each film gets a one-line "why"
     (top matched director/DP + tropes).
3. **`index.html`** is a static page that fetches `screenings.json` and
   renders the top 10 ranked films, each with its "why" line and merged
   venue showtimes.

## Setup

```bash
pip install -r requirements.txt
```

You need two API keys:

| Key | Where | Used for |
|-----|-------|----------|
| `TMDB_API_KEY` | [themoviedb.org](https://www.themoviedb.org/settings/api) — v4 read access token (recommended) or v3 API key | film metadata, posters, directors |
| `SERPAPI_API_KEY` | [serpapi.com](https://serpapi.com/) | Google showtimes per venue |

Export your Letterboxd ratings (Settings → Export) and save as `ratings.csv`
in the repo root.

```bash
# one-time / whenever ratings change
TMDB_API_KEY=... python build_taste_profile.py

# refresh this week's screenings
TMDB_API_KEY=... SERPAPI_API_KEY=... python update_screenings.py
```

For the scheduled run, add both keys as
[GitHub Actions secrets](https://docs.github.com/en/actions/security-guides/encrypted-secrets)
named `TMDB_API_KEY` and `SERPAPI_API_KEY`. GitHub Pages serves the site;
`CNAME` points it at the custom domain.

## Debugging showtime parsing

Google's showtime markup changes shape often. To capture a raw response:

1. Go to **Actions → Repertory Auto-Update → Run workflow** and tick
   **debug**.
2. When the run finishes, download the `serpapi-debug` artifact — it
   contains the raw JSON per query.
3. Adjust `extract_times` / `iter_movie_blocks` in `update_screenings.py`
   to match, then re-run.

## Project layout

| File | Purpose |
|------|---------|
| `index.html` | static site; fetches `screenings.json` |
| `screenings.json` | generated weekly screening data (do not hand-edit) |
| `update_screenings.py` | SerpApi → TMDB → scoring → `screenings.json` |
| `build_taste_profile.py` | `ratings.csv` → `taste_profile.json` |
| `tmdb_client.py` | shared TMDB client + title/summary text utils |
| `ratings.csv` | Letterboxd export (Name, Year, Rating) |
| `taste_profile.json` | generated taste model |
| `.github/workflows/` | Friday auto-update + manual profile rebuild |
