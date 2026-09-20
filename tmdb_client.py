"""Shared TMDB client + film-title text utilities.

Used by both build_taste_profile.py (one-time taste model build) and
update_screenings.py (weekly screening refresh) so the TMDB access logic
lives in exactly one place.
"""
import re
import urllib.parse

import requests

BASE_URL = "https://api.themoviedb.org/3"


def clean_film_title(raw_title):
    """Strip repertory-screening suffixes (35mm, restoration, Q&A, ...) so the
    title can be matched against TMDB and de-duplicated."""
    t = (raw_title or "").strip()
    t = re.sub(r'\(.*?\)|\[.*?\]', '', t)
    t = re.sub(
        r'\b(35mm|70mm|16mm|4k|restoration|restored|dcp|q&a|in person|'
        r'repertory|special screening|preview|staff picks|with live score|'
        r'waverly midnights|imax|rpx|3d)\b', '', t, flags=re.I)
    if " - " in t:
        t = t.split(" - ")[0]
    if " \u2013 " in t:
        t = t.split(" \u2013 ")[0]
    return re.sub(r'\s+', ' ', t).strip()


def trim_summary(text, max_chars=130):
    """Trim a synopsis to a card-friendly length, preferring a full sentence."""
    if not text:
        return ""
    text = text.strip()
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if sentences and 25 <= len(sentences[0]) <= max_chars:
        return sentences[0]
    if len(text) > max_chars:
        truncated = text[:max_chars].rsplit(' ', 1)[0]
        return truncated.rstrip('.,;:-') + '...'
    return text


class TMDBClient:
    """Thin wrapper around the TMDB v3 API with result caching.

    Auth: a v4 read-access token (JWT, starts with "eyJ") is sent as a Bearer
    header (TMDB's preferred method); a legacy v3 API key falls back to the
    api_key query parameter.
    """

    def __init__(self, api_key):
        self.api_key = (api_key or "").strip()
        self._cache = {}
        self._session = requests.Session()
        if self.api_key.startswith("eyJ"):
            self._session.headers["Authorization"] = f"Bearer {self.api_key}"
            self._use_param = False
        else:
            self._use_param = True

    @property
    def configured(self):
        return bool(self.api_key)

    def _get(self, path, params=None):
        params = dict(params or {})
        if self._use_param and self.api_key:
            params["api_key"] = self.api_key
        resp = self._session.get(BASE_URL + path, params=params, timeout=8)
        resp.raise_for_status()
        return resp.json()

    def fetch_movie(self, title, year=None):
        """Return enriched metadata for a film title, or None.

        Result keys: title, director, directors (list), dps (list),
        year, overview, corpus, poster.
        """
        clean_search = clean_film_title(title)
        if not self.configured or len(clean_search) < 2:
            return None
        cache_key = clean_search.lower()
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            params = {"query": clean_search}
            if year and str(year).isdigit():
                params["year"] = str(year)
            search = self._get("/search/movie", params)
            results = search.get("results", [])
            if not results:
                self._cache[cache_key] = None
                return None

            movie = results[0]
            movie_id = movie["id"]

            credits = self._get(f"/movie/{movie_id}/credits")
            directors = [c["name"] for c in credits.get("crew", [])
                         if c.get("job") == "Director"]
            dps = [c["name"] for c in credits.get("crew", [])
                   if c.get("job") in ("Director of Photography", "Cinematographer")]

            details = self._get(f"/movie/{movie_id}",
                                {"append_to_response": "keywords,reviews"})
            overview = details.get("overview", "")
            keywords = [k["name"].lower()
                        for k in details.get("keywords", {}).get("keywords", [])]
            reviews = [r["content"]
                       for r in details.get("reviews", {}).get("results", [])[:2]]
            release_date = movie.get("release_date", "") or ""
            year_out = (int(release_date.split("-")[0])
                        if release_date.split("-")[0].isdigit() else None)
            poster_path = movie.get("poster_path")
            poster = (f"https://image.tmdb.org/t/p/w500{poster_path}"
                      if poster_path else None)

            data = {
                "title": movie.get("title", clean_search),
                "director": directors[0] if directors else "Unknown",
                "directors": directors,
                "dps": dps,
                "year": year_out,
                "overview": overview,
                "corpus": f"{overview} {' '.join(keywords)} {' '.join(reviews)}",
                "poster": poster,
            }
            self._cache[cache_key] = data
            return data
        except Exception as exc:  # network / API hiccup -> skip enrichment
            print(f"[TMDB] Skipping '{title}': {exc}")
            self._cache[cache_key] = None
            return None
