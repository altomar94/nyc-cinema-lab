import re
import urllib.request
import xml.etree.ElementTree as ET

# 1. Configuration
LETTERBOXD_USERNAME = "TK94"
PROFILE_URL = f"https://letterboxd.com/{LETTERBOXD_USERNAME}/"
RSS_URL = f"https://letterboxd.com/{LETTERBOXD_USERNAME}/rss/"

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

# 2. Fetch Total Films Count from Profile Page
total_films = "224"  # Default fallback
try:
    req = urllib.request.Request(PROFILE_URL, headers=headers)
    with urllib.request.urlopen(req) as resp:
        profile_html = resp.read().decode('utf-8')
    
    # Scrape total count from profile stats link
    count_match = re.search(r'href="/' + re.escape(LETTERBOXD_USERNAME) + r'/films/"[^>]*>\s*<span[^>]*class="value"[^>]*>([\d,]+)</span>', profile_html)
    if not count_match:
        count_match = re.search(r'href="/' + re.escape(LETTERBOXD_USERNAME) + r'/films/"[^>]*>\s*([\d,]+)', profile_html)
        
    if count_match:
        total_films = count_match.group(1).replace(',', '')
        print(f"Fetched total logged films: {total_films}")
except Exception as e:
    print(f"Warning: Could not fetch total films count from profile ({e}).")

# 3. Fetch RSS Feed for Watched Titles & Mean Rating
watched_titles = set()
ratings = []

try:
    req = urllib.request.Request(RSS_URL, headers=headers)
    with urllib.request.urlopen(req) as resp:
        xml_data = resp.read()
    root = ET.fromstring(xml_data)
    
    for item in root.findall('./channel/item'):
        title_elem = item.find('title')
        if title_elem is not None and title_elem.text:
            text = title_elem.text
            clean_title = text.split(' - ')[0].split(', 19')[0].split(', 20')[0].strip().lower()
            watched_titles.add(clean_title)
            
            # Extract rating stars from RSS entry title
            if ' - ' in text:
                rating_str = text.split(' - ')[-1].strip()
                stars = rating_str.count('★') + (0.5 if '½' in rating_str else 0)
                if stars > 0:
                    ratings.append(stars)
                    
    print(f"Fetched {len(watched_titles)} titles and {len(ratings)} rated entries from RSS.")
except Exception as e:
    print(f"Warning: Letterboxd RSS fetch failed ({e}).")

# Calculate Mean Rating from recent logs (fallback to 3.69 if no ratings present)
mean_rating = round(sum(ratings) / len(ratings), 2) if ratings else 3.69

# 4. Read index.html
with open("index.html", "r", encoding="utf-8") as f:
    html = f.read()

# 5. Update Header Statistics in HTML
html = re.sub(
    r'<div>\d+\s*FILMS LOGGED\s*//\s*MEAN RATING:\s*[\d\.]+\s*★</div>',
    f'<div>{total_films} FILMS LOGGED // MEAN RATING: {mean_rating} ★</div>',
    html
)

html = re.sub(
    r'<span>LOGGED:\s*<strong>\d+\s*FILMS</strong></span>',
    f'<span>LOGGED: <strong>{total_films} FILMS</strong></span>',
    html
)

html = re.sub(
    r'<span>MEAN:\s*<strong>[\d\.]+\s*★</strong></span>',
    f'<span>MEAN: <strong>{mean_rating} ★</strong></span>',
    html
)

# 6. Update Screening 'seen' Badges
def update_seen_status(match):
    block = match.group(0)
    title_match = re.search(r'title:\s*["\']([^"\']+)["\']', block)
    if title_match:
        film_title = title_match.group(1).strip().lower()
        if film_title in watched_titles:
            block = re.sub(r'seen:\s*false', 'seen: true', block)
    return block

html = re.sub(r'\{\s*title:\s*["\'].*?\}', update_seen_status, html, flags=re.DOTALL)

# 7. Write Back Changes
with open("index.html", "w", encoding="utf-8") as f:
    f.write(html)

print(f"Successfully updated index.html (Logged: {total_films}, Mean: {mean_rating}★)")
