import os
import time
import io
import sqlite3
import requests
import numpy as np
import librosa
import onnxruntime as ort
import faiss
import yt_dlp

FAISS_INDEX_PATH = "models/music_ozz_faiss.index"
SQLITE_DB_PATH = "models/music_ozz_metadata.db"
ONNX_MODEL_PATH = "models/music_ozz_encoder.onnx"

PRIORITY_VIEW_THRESHOLD = 100_000
SLEEP_BETWEEN_SONGS_SEC = 1.5
SLEEP_BETWEEN_CYCLES_SEC = 900

SEARCH_QUERIES = [
    "trending songs official audio",
    "top viral hits music",
    "popular movie soundtrack theme",
    "billboard top songs audio",
    "latest official music releases"
]

class AutonomousLearner:
    def __init__(self):
        os.makedirs(os.path.dirname(SQLITE_DB_PATH) or ".", exist_ok=True)
        
        if os.path.exists(ONNX_MODEL_PATH):
            self.session = ort.InferenceSession(ONNX_MODEL_PATH, providers=["CPUExecutionProvider"])
            self.input_name = self.session.get_inputs()[0].name
        else:
            self.session = None
            self.input_name = None
            print(f"[Warning] ONNX model not found at {ONNX_MODEL_PATH}. Continuous learner idling until model is provided.")

        if os.path.exists(FAISS_INDEX_PATH):
            self.faiss_index = faiss.read_index(FAISS_INDEX_PATH)
        else:
            base_index = faiss.IndexFlatIP(128)
            self.faiss_index = faiss.IndexIDMap2(base_index)

    def extract_vector(self, audio_bytes: bytes) -> np.ndarray:
        if not self.session:
            raise RuntimeError("ONNX model session not initialized.")
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=22050, duration=10.0, mono=True)
        if len(y) == 0:
            raise ValueError("Empty audio stream.")
            
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        norm_spec = (log_mel_spec - np.mean(log_mel_spec)) / (np.std(log_mel_spec) + 1e-6)
        tensor = norm_spec[np.newaxis, np.newaxis, :, :].astype(np.float32)

        outputs = self.session.run(None, {self.input_name: tensor})
        embedding = outputs[0]
        faiss.normalize_L2(embedding)
        return embedding

    def save_vector(self, song_id: int, embedding: np.ndarray):
        self.faiss_index.add_with_ids(embedding, np.array([song_id], dtype=np.int64))
        faiss.write_index(self.faiss_index, FAISS_INDEX_PATH)

def fetch_all_candidate_tracks(queries: list):
    ydl_opts = {'quiet': True, 'extract_flat': 'in_playlist', 'skip_download': True}
    all_candidates = []
    seen_urls = set()

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        for query in queries:
            try:
                info = ydl.extract_info(f"ytsearch25:{query}", download=False)
                for entry in info.get('entries', []):
                    video_id = entry.get('id')
                    if not video_id:
                        continue
                    url = f"https://www.youtube.com/watch?v={video_id}"
                    if url in seen_urls:
                        continue
                    seen_urls.add(url)
                    view_count = entry.get('view_count') or 0

                    all_candidates.append({
                        "url": url,
                        "title": entry.get("title", "Unknown Track"),
                        "artist": entry.get("uploader", "Unknown Artist"),
                        "view_count": view_count
                    })
            except Exception as e:
                print(f"⚠️️ Search error for query '{query}': {e}")

    all_candidates.sort(key=lambda x: x["view_count"], reverse=True)
    return all_candidates

def get_audio_stream(video_url: str) -> bytes:
    ydl_opts = {'format': 'bestaudio/best', 'quiet': True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=False)
        audio_url = info.get('url')
        if audio_url:
            res = requests.get(audio_url, stream=True, timeout=10)
            return res.content
    raise ValueError("Audio stream unavailable.")

def run_autonomous_loop():
    print("🤖 Music Ozz 24/7 Priority Autonomous Learner Started...")
    learner = AutonomousLearner()

    while True:
        conn = sqlite3.connect(SQLITE_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS songs (
                song_id INTEGER PRIMARY KEY,
                title TEXT NOT NULL,
                artist TEXT NOT NULL,
                view_count INTEGER,
                youtube_url TEXT UNIQUE,
                file_path TEXT
            )
        """)
        conn.commit()

        if not learner.session:
            print("⚠️ Waiting for ONNX encoder model... Idle loop active.")
            conn.close()
            time.sleep(SLEEP_BETWEEN_CYCLES_SEC)
            continue

        print(f"\n🔍 Crawling tracks across {len(SEARCH_QUERIES)} search queries...")
        candidates = fetch_all_candidate_tracks(SEARCH_QUERIES)

        priority_queue = [t for t in candidates if t["view_count"] >= PRIORITY_VIEW_THRESHOLD]
        secondary_queue = [t for t in candidates if t["view_count"] < PRIORITY_VIEW_THRESHOLD]

        full_queue = priority_queue + secondary_queue

        for idx, track in enumerate(full_queue, start=1):
            cursor.execute("SELECT song_id FROM songs WHERE youtube_url = ?", (track["url"],))
            if cursor.fetchone():
                continue

            category = "🔥 PRIORITY (>100k)" if track["view_count"] >= PRIORITY_VIEW_THRESHOLD else "🎵 SECONDARY (<100k)"

            try:
                print(f"[{idx}/{len(full_queue)}] {category} [{track['view_count']:,} views]: {track['artist']} - {track['title']}")
                audio_bytes = get_audio_stream(track["url"])
                embedding = learner.extract_vector(audio_bytes)

                cursor.execute("SELECT COALESCE(MAX(song_id), 100000) + 1 FROM songs")
                new_id = cursor.fetchone()[0]

                cursor.execute(
                    "INSERT INTO songs (song_id, title, artist, view_count, youtube_url, file_path) VALUES (?, ?, ?, ?, ?, ?)",
                    (new_id, track["title"], track["artist"], track["view_count"], track["url"], "auto_learned")
                )
                conn.commit()

                learner.save_vector(new_id, embedding)
                print(f"   ↳ ✅ Learned into FAISS [ID: {new_id}]")
                time.sleep(SLEEP_BETWEEN_SONGS_SEC)

            except Exception as e:
                print(f"   ↳ ⚠️ Skipping track: {e}")

        conn.close()
        print(f"\n💤 Sweep complete. Total vectors in FAISS: {learner.faiss_index.ntotal}")
        time.sleep(SLEEP_BETWEEN_CYCLES_SEC)

if __name__ == "__main__":
    run_autonomous_loop()
    
