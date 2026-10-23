import time
import sqlite3
import requests
import deezer

DB_PATH = "models/music_ozz_metadata.db"
INDEX_URL = "http://localhost:8000/api/v1/songs/index"
SLEEP_BETWEEN_SONGS_SEC = 1.5
SLEEP_BETWEEN_CYCLES_SEC = 900

def already_learned(title, artist):
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute("SELECT 1 FROM songs WHERE title = ? AND artist = ?", (title, artist)).fetchone()
    conn.close()
    return row is not None

def run():
    time.sleep(30)
    while True:
        for track in deezer.trending():
            title, artist = track["title"], track["artist"]["name"]
            if not track.get("preview") or already_learned(title, artist):
                continue
            try:
                audio = requests.get(track["preview"], timeout=15).content
                requests.post(
                    INDEX_URL,
                    data={"title": title, "artist": artist},
                    files={"file": ("preview.mp3", audio, "audio/mpeg")},
                    timeout=60,
                )
                print(f"Learned: {artist} - {title}")
            except Exception as e:
                print(f"Skipped {title}: {e}")
            time.sleep(SLEEP_BETWEEN_SONGS_SEC)
        time.sleep(SLEEP_BETWEEN_CYCLES_SEC)

if __name__ == "__main__":
    run()
    
