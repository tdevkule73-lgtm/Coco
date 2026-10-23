import requests

API = "https://api.deezer.com"

def trending():
    seen, out = set(), []
    for g in requests.get(f"{API}/genre", timeout=10).json()["data"]:
        r = requests.get(f"{API}/chart/{g['id']}/tracks", params={"limit": 50}, timeout=10).json()
        for t in r.get("data", []):
            if t["id"] not in seen:
                seen.add(t["id"])
                out.append(t)
    return out

def preview_bytes(title, artist=""):
    q = f'track:"{title}" artist:"{artist}"'
    data = requests.get(f"{API}/search", params={"q": q, "limit": 1}, timeout=10).json().get("data", [])
    if data and data[0].get("preview"):
        return requests.get(data[0]["preview"], timeout=15).content
      
