import re
import urllib.request
import xml.etree.ElementTree as ET

# 1. Configuration
LETTERBOXD_USERNAME = "TK94"
RSS_URL = "https://letterboxd.com/tk94/rss/"

# 2. Fetch Watched Titles from Letterboxd RSS
watched_titles = set()
req = urllib.request.Request(RSS_URL, headers={'User-Agent': 'Mozilla/5.0'})

try:
    with urllib.request.urlopen(req) as response:
        xml_data = response.read()
    root = ET.fromstring(xml_data)
    for item in root.findall('./channel/item'):
        title_elem = item.find('title')
        if title_elem is not None and title_elem.text:
            clean_title = title_elem.text.split(' - ')[0].split(', 19')[0].split(', 20')[0].strip().lower()
            watched_titles.add(clean_title)
    print(f"Successfully fetched {len(watched_titles)} titles from Letterboxd.")
except Exception as e:
    print(f"Warning: Letterboxd RSS fetch failed ({e}). Proceeding without updating seen statuses.")

# 3. Read index.html
with open("index.html", "r", encoding="utf-8") as f:
    html = f.read()

# 4. Safely update JavaScript object blocks without JSON decoding
def update_seen_status(match):
    block = match.group(0)
    title_match = re.search(r'title:\s*["\']([^"\']+)["\']', block)
    if title_match:
        film_title = title_match.group(1).strip().lower()
        if film_title in watched_titles:
            block = re.sub(r'seen:\s*false', 'seen: true', block)
    return block

updated_html = re.sub(r'\{\s*title:\s*["\'].*?\}', update_seen_status, html, flags=re.DOTALL)

# 5. Write back to index.html
with open("index.html", "w", encoding="utf-8") as f:
    f.write(updated_html)

print("Updated index.html successfully.")
