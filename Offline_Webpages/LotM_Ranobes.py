import os
import requests
from bs4 import BeautifulSoup

# URLs
chapter_list_url = "https://ranobes.top/chapters/133485/"
chapter_base_url = "https://ranobes.top/lord-of-the-mysteries-v812312-133485/"

# Fake browser headers
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/123.0.0.0 Safari/537.36"
}

# Config
start_chapter_id = 185169
total_chapters = 20  # Adjust as needed
output_file = "Lord_of_the_Mysteries_Offline.html"

# HTML template start
html_content = [
    "<html><head><meta charset='UTF-8'><title>Lord of the Mysteries</title></head><body>",
    "<h1>Lord of the Mysteries - Offline Version</h1>",
    "<h2>Table of Contents</h2><ul>"
]

# Step 1: Download chapter list (to get titles)
print("Fetching chapter list...")
resp = requests.get(chapter_list_url, headers=headers)
soup = BeautifulSoup(resp.text, "html.parser")

# Get chapter titles from the list page
chapter_links = soup.select(".chapter-list a")
chapter_titles = [a.text.strip() for a in chapter_links]

# Build Table of Contents
for i, title in enumerate(chapter_titles[:total_chapters]):
    html_content.append(f"<li><a href='#chapter{i+1}'>{title}</a></li>")
html_content.append("</ul><hr>")

# Step 2: Download and embed each chapter
for i in range(total_chapters):
    chap_id = start_chapter_id + i
    url = f"{chapter_base_url}{chap_id}.html"
    print(f"Downloading Chapter {i+1} from {url}")
    
    try:
        res = requests.get(url, headers=headers)
        res.raise_for_status()
    except requests.RequestException:
        print(f"⚠️ Failed to download chapter {i+1}")
        continue

    chap_soup = BeautifulSoup(res.text, "html.parser")
    
    # Extract content div
    content_div = chap_soup.find("div", class_="chapter-inner chapter-content")
    if not content_div:
        print(f"⚠️ Chapter content not found for chapter {i+1}")
        continue

    # Add chapter title and content
    title = chapter_titles[i] if i < len(chapter_titles) else f"Chapter {i+1}"
    html_content.append(f"<h2 id='chapter{i+1}'>{title}</h2>")
    html_content.append(str(content_div))
    html_content.append("<hr>")

# Wrap up HTML
html_content.append("</body></html>")

# Step 3: Write to file
with open(output_file, "w", encoding="utf-8") as f:
    f.write("\n".join(html_content))

print(f"\n✅ Offline book created: {output_file}")
