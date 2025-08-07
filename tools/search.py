import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, parse_qs, unquote

def free_search(query, num=2):
    headers = {"User-Agent": "Mozilla/5.0"}
    url = f"https://duckduckgo.com/html/?q={query}"
    resp = requests.get(url, headers=headers, timeout=10)
    soup = BeautifulSoup(resp.text, "html.parser")
    
    results = []
    for link in soup.select(".result__a")[:num]:
        href = link.get("href")
        
        # If it's a DuckDuckGo redirect link, extract the real URL
        if href.startswith("//duckduckgo.com/l/"):
            parsed_qs = parse_qs(urlparse(href).query)
            if "uddg" in parsed_qs:
                href = unquote(parsed_qs["uddg"][0])
        
        results.append(href)
    
    return results