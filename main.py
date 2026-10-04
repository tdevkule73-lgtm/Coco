import os
import io
import json
import time
import sqlite3
import hashlib
import asyncio
import requests
import numpy as np
import librosa
import onnxruntime as ort
import faiss
import redis.asyncio as redis
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any

from fastapi import FastAPI, File, UploadFile, HTTPException, Response, Request, status, BackgroundTasks, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from google import genai
from google.genai import types

MAX_FILE_SIZE = 15 * 1024 * 1024
ALLOWED_MIME_TYPES = {"audio/wav", "audio/mpeg", "audio/x-m4a", "audio/ogg", "audio/webm", "audio/flac"}
SIMILARITY_THRESHOLD = 0.55
CACHE_TTL_SECONDS = 86400

ONNX_MODEL_PATH = os.getenv("ONNX_MODEL_PATH", "models/music_ozz_encoder.onnx")
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "models/music_ozz_faiss.index")
SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH", "models/music_ozz_metadata.db")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID", "")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET", "")

class SongMetadata(BaseModel):
    song_id: Optional[int] = Field(None, example=100001)
    title: str = Field(..., example="Avengers Theme")
    artist: str = Field(..., example="Alan Silvestri")
    album: Optional[str] = Field("Single", example="The Avengers OST")
    cover_art_url: Optional[str] = Field(None)
    spotify_url: Optional[str] = Field(None)
    youtube_url: Optional[str] = Field(None)
    source: str = Field("local", example="local")

class MatchResponse(BaseModel):
    matched: bool
    confidence_score: float = Field(..., example=0.88)
    song: Optional[SongMetadata] = None
    message: str = Field(..., example="Song identified successfully!")

def get_audio_stream_from_url(video_url: str) -> bytes:
    import yt_dlp
    ydl_opts = {'format': 'bestaudio/best', 'quiet': True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=False)
        audio_url = info.get('url')
        if audio_url:
            res = requests.get(audio_url, stream=True, timeout=15)
            return res.content
    raise ValueError("Audio stream unavailable from provided URL.")

class InternetAutoLearner:
    @staticmethod
    def get_spotify_token() -> Optional[str]:
        if not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET:
            return None
        try:
            res = requests.post(
                "https://accounts.spotify.com/api/token",
                data={"grant_type": "client_credentials"},
                auth=(SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET),
                timeout=5
            )
            if res.status_code == 200:
                return res.json().get("access_token")
        except Exception as e:
            print(f"[Warning] Spotify Auth Error: {e}")
        return None

    @classmethod
    def fetch_spotify_details(cls, title: str, artist: str) -> Dict[str, Any]:
        token = cls.get_spotify_token()
        details = {"album": "Single", "cover_art_url": None, "spotify_url": None, "preview_url": None}
        if not token:
            return details

        try:
            query = f"track:{title} artist:{artist}"
            headers = {"Authorization": f"Bearer {token}"}
            res = requests.get("https://api.spotify.com/v1/search", headers=headers, params={"q": query, "type": "track", "limit": 1}, timeout=5)
            if res.status_code == 200:
                items = res.json().get("tracks", {}).get("items", [])
                if items:
                    track = items[0]
                    images = track.get("album", {}).get("images", [])
                    details["album"] = track.get("album", {}).get("name", "Single")
                    details["cover_art_url"] = images[0]["url"] if images else None
                    details["spotify_url"] = track.get("external_urls", {}).get("spotify")
                    details["preview_url"] = track.get("preview_url")
        except Exception as e:
            print(f"[Warning] Spotify Search Error: {e}")
        return details

    @staticmethod
    def generate_youtube_search_url(title: str, artist: str) -> str:
        query = requests.utils.quote(f"{artist} {title}")
        return f"https://music.youtube.com/search?q={query}"

class SearchEngine:
    def __init__(self, onnx_path: str = ONNX_MODEL_PATH, faiss_index_path: str = FAISS_INDEX_PATH, sqlite_db_path: str = SQLITE_DB_PATH):
        self.onnx_path = onnx_path
        self.faiss_index_path = faiss_index_path
        self.sqlite_db_path = sqlite_db_path

        self.ensure_db_and_index()

        if os.path.exists(self.onnx_path):
            self.ort_session = ort.InferenceSession(self.onnx_path, providers=["CPUExecutionProvider"])
            self.input_name = self.ort_session.get_inputs()[0].name
        else:
            self.ort_session = None
            self.input_name = None
            print(f"[Warning] ONNX model not found at {self.onnx_path}. Running in Gemini-only fallback mode.")

        self.faiss_index = faiss.read_index(self.faiss_index_path)
        self.gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

    def ensure_db_and_index(self):
        os.makedirs(os.path.dirname(self.sqlite_db_path) or ".", exist_ok=True)
        conn = sqlite3.connect(self.sqlite_db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS songs (
                song_id INTEGER PRIMARY KEY,
                title TEXT NOT NULL,
                artist TEXT NOT NULL,
                album TEXT,
                cover_art_url TEXT,
                spotify_url TEXT,
                youtube_url TEXT,
                view_count INTEGER,
                file_path TEXT
            )
        """)
        conn.commit()
        conn.close()

        if not os.path.exists(self.faiss_index_path):
            base_index = faiss.IndexFlatIP(128)
            idx = faiss.IndexIDMap2(base_index)
            faiss.write_index(idx, self.faiss_index_path)

    def process_audio(self, audio_bytes: bytes) -> np.ndarray:
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=22050, duration=10.0, mono=True)
        if len(y) == 0:
            raise ValueError("Empty audio recording.")
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        norm_spec = (log_mel_spec - np.mean(log_mel_spec)) / (np.std(log_mel_spec) + 1e-6)
        return norm_spec[np.newaxis, np.newaxis, :, :].astype(np.float32)

    def run_inference(self, input_tensor: np.ndarray) -> np.ndarray:
        if not self.ort_session:
            raise RuntimeError("ONNX model session not initialized.")
        outputs = self.ort_session.run(None, {self.input_name: input_tensor})
        return outputs[0]

    def search_faiss(self, embedding: np.ndarray):
        faiss.normalize_L2(embedding)
        distances, indices = self.faiss_index.search(embedding, 1)
        return int(indices[0][0]), float(distances[0][0])

    def get_song_metadata(self, song_id: int) -> Optional[SongMetadata]:
        conn = sqlite3.connect(self.sqlite_db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT song_id, title, artist, album, cover_art_url, spotify_url, youtube_url FROM songs WHERE song_id = ?", (song_id,))
        row = cursor.fetchone()
        conn.close()
        if not row:
            return None
        return SongMetadata(
            song_id=row[0], title=row[1], artist=row[2],
            album=row[3] or "Single", cover_art_url=row[4],
            spotify_url=row[5], youtube_url=row[6], source="local"
        )

    def save_song_to_local(self, song_id: int, title: str, artist: str, audio_bytes: Optional[bytes] = None, song_url: str = "") -> SongMetadata:
        conn = sqlite3.connect(self.sqlite_db_path)
        cursor = conn.cursor()
        yt_url = song_url or InternetAutoLearner.generate_youtube_search_url(title, artist)

        cursor.execute(
            "INSERT OR REPLACE INTO songs (song_id, title, artist, youtube_url, file_path) VALUES (?, ?, ?, ?, ?)",
            (song_id, title, artist, yt_url, "url_indexed" if song_url else "local")
        )
        conn.commit()
        conn.close()

        if audio_bytes and self.ort_session:
            try:
                input_tensor = self.process_audio(audio_bytes)
                embedding = self.run_inference(input_tensor)
                faiss.normalize_L2(embedding)
                self.faiss_index.add_with_ids(embedding, np.array([song_id], dtype=np.int64))
                faiss.write_index(self.faiss_index, self.faiss_index_path)
            except Exception as e:
                print(f"[Error] FAISS indexing error: {e}")

        return SongMetadata(
            song_id=song_id, title=title, artist=artist,
            album="Single", youtube_url=yt_url, source="auto_learned"
        )

def background_internet_enrichment_task(song_id: int, title: str, artist: str):
    spotify_data = InternetAutoLearner.fetch_spotify_details(title, artist)
    yt_url = InternetAutoLearner.generate_youtube_search_url(title, artist)

    conn = sqlite3.connect(SQLITE_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE songs 
        SET album = COALESCE(?, album),
            cover_art_url = COALESCE(?, cover_art_url),
            spotify_url = COALESCE(?, spotify_url),
            youtube_url = COALESCE(?, youtube_url)
        WHERE song_id = ?
    """, (spotify_data["album"], spotify_data["cover_art_url"], spotify_data["spotify_url"], yt_url, song_id))
    conn.commit()
    conn.close()

    if spotify_data["preview_url"] and search_engine and search_engine.ort_session:
        try:
            res = requests.get(spotify_data["preview_url"], timeout=10)
            if res.status_code == 200:
                audio_bytes = res.content
                input_tensor = search_engine.process_audio(audio_bytes)
                embedding = search_engine.run_inference(input_tensor)
                faiss.normalize_L2(embedding)
                search_engine.faiss_index.add_with_ids(embedding, np.array([song_id], dtype=np.int64))
                faiss.write_index(search_engine.faiss_index, FAISS_INDEX_PATH)
        except Exception as e:
            print(f"Studio preview learning error: {e}")

search_engine: Optional[SearchEngine] = None
redis_client: Optional[redis.Redis] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global search_engine, redis_client
    os.makedirs("static", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    search_engine = SearchEngine()
    redis_client = redis.Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", 6379)),
        db=0, decode_responses=True, socket_timeout=3.0
    )
    yield
    try:
        await redis_client.aclose()
    except Exception:
        pass

app = FastAPI(title="Music Ozz API", version="1.9.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=FileResponse)
async def serve_index():
    return FileResponse("static/index.html")

@app.post("/api/v1/identify", response_model=MatchResponse)
async def identify_song(background_tasks: BackgroundTasks, response: Response, file: UploadFile = File(...)):
    audio_bytes = await file.read()
    if len(audio_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large.")

    audio_hash = hashlib.sha256(audio_bytes).hexdigest()
    cache_key = f"music_ozz:audio:{audio_hash}"

    try:
        cached_result = await redis_client.get(cache_key)
        if cached_result:
            response.headers["X-Cache"] = "HIT"
            return MatchResponse(**json.loads(cached_result))
    except Exception:
        pass

    response.headers["X-Cache"] = "MISS"

    if search_engine and search_engine.ort_session:
        try:
            input_tensor = await asyncio.to_thread(search_engine.process_audio, audio_bytes)
            embedding = await asyncio.to_thread(search_engine.run_inference, input_tensor)
            song_id, confidence = await asyncio.to_thread(search_engine.search_faiss, embedding)
            song_info = await asyncio.to_thread(search_engine.get_song_metadata, song_id)

            if confidence >= SIMILARITY_THRESHOLD and song_info is not None:
                match_response = MatchResponse(
                    matched=True,
                    confidence_score=round(confidence, 4),
                    song=song_info,
                    message="Matched from local memory!"
                )
                try:
                    await redis_client.setex(cache_key, CACHE_TTL_SECONDS, match_response.model_dump_json())
                except Exception:
                    pass
                return match_response
        except Exception as e:
            print(f"[Info] Local search skipped/failed: {e}")

    mime_type = file.content_type if file.content_type in ALLOWED_MIME_TYPES else "audio/webm"
    
    if search_engine and search_engine.gemini_client:
        prompt = (
            "Listen to this audio. Identify the song title and artist or theme name. "
            "Respond strictly with valid JSON: "
            '{"matched": true, "title": "Song Title", "artist": "Artist Name", "confidence_score": 0.85}'
            "If no match: "
            '{"matched": false, "title": "", "artist": "", "confidence_score": 0.0}'
        )
        try:
            gem_res = search_engine.gemini_client.models.generate_content(
                model='gemini-2.5-flash',
                contents=[types.Part.from_bytes(data=audio_bytes, mime_type=mime_type), prompt],
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            res_data = json.loads(gem_res.text)

            if res_data.get("matched"):
                title = res_data.get("title", "Unknown Title")
                artist = res_data.get("artist", "Unknown Artist")

                conn = sqlite3.connect(SQLITE_DB_PATH)
                cursor = conn.cursor()
                cursor.execute("SELECT COALESCE(MAX(song_id), 100000) + 1 FROM songs")
                new_song_id = cursor.fetchone()[0]
                conn.close()

                saved_song = search_engine.save_song_to_local(new_song_id, title, artist, audio_bytes)
                background_tasks.add_task(background_internet_enrichment_task, new_song_id, title, artist)

                match_resp = MatchResponse(
                    matched=True,
                    confidence_score=float(res_data.get("confidence_score", 0.85)),
                    song=saved_song,
                    message="Identified via Google Gemini & auto-learned to local memory!"
                )
                try:
                    await redis_client.setex(cache_key, CACHE_TTL_SECONDS, match_resp.model_dump_json())
                except Exception:
                    pass
                return match_resp
        except Exception as e:
            print(f"[Error] Gemini API Error: {e}")

    return MatchResponse(matched=False, confidence_score=0.0, song=None, message="No matching song found.")

@app.post("/api/v1/songs/index")
async def upload_and_index_song(
    file: Optional[UploadFile] = File(None),
    song_url: Optional[str] = Form(None),
    title: str = Form(...),
    artist: str = Form("Unknown Artist"),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    if not file and not song_url:
        raise HTTPException(status_code=400, detail="Provide either a song URL or an audio file.")

    audio_bytes = None

    if song_url and song_url.strip():
        try:
            audio_bytes = await asyncio.to_thread(get_audio_stream_from_url, song_url.strip())
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to fetch audio from link: {str(e)}")
    elif file:
        audio_bytes = await file.read()

    if not audio_bytes or len(audio_bytes) == 0:
        raise HTTPException(status_code=400, detail="Audio content is empty.")

    try:
        conn = sqlite3.connect(SQLITE_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT COALESCE(MAX(song_id), 100000) + 1 FROM songs")
        new_song_id = cursor.fetchone()[0]
        conn.close()

        saved_song = search_engine.save_song_to_local(new_song_id, title.strip(), artist.strip(), audio_bytes, song_url or "")
        background_tasks.add_task(background_internet_enrichment_task, new_song_id, title.strip(), artist.strip())

        return {"status": "success", "message": f"Successfully learned '{title}' into AI memory!", "song_id": new_song_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to index song: {str(e)}")

@app.post("/api/v1/songs/enroll-hum")
async def enroll_user_hum(file: UploadFile = File(...), song_id: int = Form(...)):
    audio_bytes = await file.read()
    if len(audio_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty audio recording.")

    conn = sqlite3.connect(SQLITE_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT title, artist FROM songs WHERE song_id = ?", (song_id,))
    song = cursor.fetchone()
    conn.close()

    if not song:
        raise HTTPException(status_code=404, detail="Song ID not found.")

    if not (search_engine and search_engine.ort_session):
        raise HTTPException(status_code=400, detail="Local encoder model unavailable.")

    try:
        input_tensor = await asyncio.to_thread(search_engine.process_audio, audio_bytes)
        embedding = await asyncio.to_thread(search_engine.run_inference, input_tensor)
        faiss.normalize_L2(embedding)
        search_engine.faiss_index.add_with_ids(embedding, np.array([song_id], dtype=np.int64))
        faiss.write_index(search_engine.faiss_index, FAISS_INDEX_PATH)

        return {"status": "success", "message": f"Trained voice print for '{song[0]}'", "song_id": song_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to enroll voice recording: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
            
