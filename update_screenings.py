import re

# Insert your weekly scraped dataset or API response here
fresh_dataset_json = """[
  {
    "title": "Le Samourai",
    "director": "Jean-Pierre Melville",
    "year": 1967,
    "theater": "The Paris Theater",
    "neighborhood": "Midtown",
    "matchScore": 94,
    "seen": false,
    "weekend": "current",
    "summary": "A methodical Parisian hitman executes a contract with icy precision.",
    "format": "4K Restoration",
    "showtimes": ["Fri: 8:00 PM", "Sat: 6:00 PM"],
    "svg": `<svg viewBox="0 0 200 300">...</svg>`
  }
]"""

with open("index.html", "r") as f:
    html = f.read()

updated_html = re.sub(
    r"const dataset = \[.*?\];",
    f"const dataset = {fresh_dataset_json};",
    html,
    flags=re.DOTALL,
)

with open("index.html", "w") as f:
    f.write(updated_html)
