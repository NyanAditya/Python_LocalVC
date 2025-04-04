import os
import requests
from bs4 import BeautifulSoup

# Base URLs
chapter_list_url = "https://ranobes.top/chapters/133485/"
chapter_base_url = "https://ranobes.top/lord-of-the-mysteries-v812312-133485/"

# HTTP headers to avoid 403
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/123.0.0.0 Safari/537.36"
}

# Settings
start_chapter_id = 185169
total_chapters = 20
output_folder = "Lord_of_the_Mysteries"

# Create output folder
os.makedirs(output_folder, exist_ok=True)

# Save the chapter list page
print("Downloading chapter list...")
resp = requests.get(chapter_list_url, headers=headers)
with open(os.path.join(output_folder, "chapter_list.html"), "w", encoding="utf-8") as f:
    f.write(resp.text)

# Function to download a single chapter
def download_chapter(chap_id, chap_num):
    url = f"{chapter_base_url}{chap_id}.html"
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        filename = f"chapter_{chap_num:04d}.html"
        with open(os.path.join(output_folder, filename), "w", encoding="utf-8") as f:
            f.write(response.text)
        print(f"✓ Downloaded: Chapter {chap_num}")
    else:
        print(f"✗ Failed: Chapter {chap_num} (ID: {chap_id}) - Status {response.status_code}")

# Download all chapters
for i in range(total_chapters):
    chapter_id = start_chapter_id + i
    download_chapter(chapter_id, i + 1)

print("✅ All downloads complete!")
