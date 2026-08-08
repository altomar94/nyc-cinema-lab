import re
import urllib.request
import xml.etree.ElementTree as ET
import json

# 1. Configuration
LETTERBOXD_USERNAME = "TK94" # Replace with your actual handle
RSS_URL = "https://letterboxd.com/TK94/rss/"

# 2. Fetch and Parse Letterboxd RSS Feed
req = urllib.request.Request(RSS_URL, headers={'User-Agent': 'Mozilla/5.0'})
try:
    with urllib.request.urlopen(req) as response:
        xml_data = response.read()
    
    root = ET.fromstring(xml_data)
    items = root.findall('./channel/item')
    
    watched_titles = set()
    for item in items:
        title_elem = item.find('title')
        if title_elem is not None and title_elem.text:
            # RSS titles are formatted as "Film Title, Year - ★★★★"
            clean_title = title_elem.text.split(' - ')[0].split(', 19')[0].split(', 20')[0].strip().lower()
            watched_titles.add(clean_title)
            
except Exception as e:
    print(f"Error fetching Letterboxd RSS feed: {e}")
    watched_titles = set()

# 3. Read Current index.html
with open("index.html", "r", encoding="utf-8") as f:
    html = f.read()

# 4. Extract Existing Dataset Array
match = re.search(r"const dataset = (\[.*?\]);", html, re.DOTALL)
if match:
    dataset = json.loads(match.group(1))
    
    # Cross-reference screenings against Letterboxd watched titles
    for film in dataset:
        if film["title"].lower() in watched_titles:
            film["seen"] = True
            
    # Serialize back to JSON string
    updated_dataset_json = json.dumps(dataset, indent=6)
    
    # Replace dataset in HTML
    html = re.sub(
        r"const dataset = \[.*?\];",
        f"const dataset = {updated_dataset_json};",
        html,
        flags=re.DOTALL
    )

# 5. Save Updated index.html
with open("index.html", "w", encoding="utf-8") as f:
    f.write(html)

print("Successfully synced Letterboxd watched status with index.html")
