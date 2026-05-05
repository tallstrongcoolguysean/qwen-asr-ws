"""Qwen3-ASR WebSocket streaming server.

Mirrors the JSON protocol used by the existing Riva proxy:
  Client -> server: {type: "config", config: {language, sampleRate}}
                    {type: "audio", audio: <base64 PCM int16>}
                    {type: "done"}
  Server -> client: {type: "ready"}
                    {type: "transcription", results: [{is_final, alternatives:[{transcript, words}]}]}
                    {type: "done"}
                    {type: "error", error}
"""

import asyncio
import base64
import json
import logging
import os
import time
from typing import Optional

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from qwen_asr import Qwen3ASRModel

# -------------------- config --------------------

MODEL_PATH = os.environ.get("MODEL_PATH", "Qwen/Qwen3-ASR-0.6B")
PORT = int(os.environ.get("PORT", "8765"))
HOST = os.environ.get("HOST", "0.0.0.0")
DEVICE_MAP = os.environ.get("DEVICE_MAP", "cuda:0")  # "cpu" / "mps" / "cuda:0"
DTYPE = os.environ.get("DTYPE", "bfloat16")          # "float32" on CPU/MPS, "bfloat16" on CUDA
SAMPLE_RATE_DEFAULT = 16000
TRANSCRIBE_INTERVAL = float(os.environ.get("TRANSCRIBE_INTERVAL", "0.7"))
MAX_BUFFER_SECONDS = int(os.environ.get("MAX_BUFFER_SECONDS", "30"))
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")

DTYPE_MAP = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}

# BCP-47 -> qwen-asr language string
LANGUAGE_MAP = {
    "de": "German", "de-DE": "German",
    "nl": "Dutch", "nl-NL": "Dutch", "nl-BE": "Dutch",
    "sv": "Swedish", "sv-SE": "Swedish",
    "el": "Greek", "el-GR": "Greek",
    "en": "English", "en-US": "English", "en-GB": "English",
    "es": "Spanish", "es-ES": "Spanish", "es-US": "Spanish",
    "fr": "French", "fr-FR": "French", "fr-CA": "French",
    "it": "Italian", "it-IT": "Italian",
    "pt": "Portuguese", "pt-BR": "Portuguese", "pt-PT": "Portuguese",
    "ar": "Arabic", "ar-AR": "Arabic",
    "zh": "Chinese", "zh-CN": "Chinese",
    "ja": "Japanese", "ja-JP": "Japanese",
    "ko": "Korean", "ko-KR": "Korean",
    "ru": "Russian", "ru-RU": "Russian",
    "hi": "Hindi", "hi-IN": "Hindi",
    "id": "Indonesian", "id-ID": "Indonesian",
    "fil": "Filipino",
    "fa": "Persian", "fa-IR": "Persian",
    "tr": "Turkish", "tr-TR": "Turkish",
    "pl": "Polish", "pl-PL": "Polish",
    "cs": "Czech", "cs-CZ": "Czech",
    "da": "Danish", "da-DK": "Danish",
    "fi": "Finnish", "fi-FI": "Finnish",
    "th": "Thai", "th-TH": "Thai",
    "vi": "Vietnamese", "vi-VN": "Vietnamese",
    "hu": "Hungarian",
    "ro": "Romanian",
    "mk": "Macedonian",
    "ms": "Malay",
}

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("qwen-asr-server")

model: Optional[Qwen3ASRModel] = None
active_sessions = 0


# -------------------- model loading --------------------

app = FastAPI()


@app.on_event("startup")
async def load_model():
    global model
    log.info(f"Loading {MODEL_PATH} on {DEVICE_MAP} ({DTYPE})...")
    t0 = time.time()
    model = Qwen3ASRModel.from_pretrained(
        MODEL_PATH,
        dtype=DTYPE_MAP[DTYPE],
        device_map=DEVICE_MAP,
        max_inference_batch_size=32,
        max_new_tokens=256,
    )
    log.info(f"Model loaded in {time.time() - t0:.1f}s")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": MODEL_PATH,
        "loaded": model is not None,
        "active_sessions": active_sessions,
    }


# -------------------- per-session state --------------------

class StreamSession:
    def __init__(self, language: str, sample_rate: int):
        self.language = language
        self.sample_rate = sample_rate
        self.audio_buffer = bytearray()
        self.last_full_text = ""
        self.committed_text = ""
        self.running = True

    def append(self, pcm: bytes):
        self.audio_buffer.extend(pcm)
        max_bytes = MAX_BUFFER_SECONDS * self.sample_rate * 2
        if len(self.audio_buffer) > max_bytes:
            del self.audio_buffer[: len(self.audio_buffer) - max_bytes]

    def audio_array(self) -> np.ndarray:
        if not self.audio_buffer:
            return np.zeros(0, dtype=np.float32)
        return np.frombuffer(bytes(self.audio_buffer), dtype=np.int16).astype(np.float32) / 32768.0

    def stop(self):
        self.running = False


def stable_prefix(prev: str, curr: str, after: str) -> str:
    """LocalAgreement-2: word-level longest common prefix between two transcripts,
    starting after the already-committed text."""
    if not prev.startswith(after) or not curr.startswith(after):
        return after
    prev_tail = prev[len(after):].split()
    curr_tail = curr[len(after):].split()
    common = []
    for a, b in zip(prev_tail, curr_tail):
        if a == b:
            common.append(a)
        else:
            break
    if not common:
        return after
    return after + (" " if after and not after.endswith(" ") else "") + " ".join(common)


# -------------------- transcription loop --------------------

async def run_transcribe(session: StreamSession) -> str:
    """Run one transcribe call in a thread (GPU work is blocking)."""
    audio = session.audio_array()
    if len(audio) < session.sample_rate * 0.3:
        return ""

    def _call():
        results = model.transcribe(
            audio=(audio, session.sample_rate),
            language=session.language,
            return_time_stamps=False,
        )
        return results[0].text if results else ""

    return await asyncio.to_thread(_call)


async def transcribe_loop(ws: WebSocket, session: StreamSession):
    """Periodic transcribe + LocalAgreement emission."""
    while session.running:
        await asyncio.sleep(TRANSCRIBE_INTERVAL)
        if not session.running:
            break

        try:
            text = await run_transcribe(session)
        except Exception as e:
            log.exception("transcribe failed")
            await safe_send(ws, {"type": "error", "error": f"transcribe: {e}"})
            continue

        if not text or text == session.last_full_text:
            continue

        new_committed = stable_prefix(session.last_full_text, text, session.committed_text)
        if len(new_committed) > len(session.committed_text):
            final_chunk = new_committed[len(session.committed_text):].strip()
            if final_chunk:
                await safe_send(ws, {
                    "type": "transcription",
                    "results": [{
                        "is_final": True,
                        "alternatives": [{"transcript": final_chunk, "words": []}],
                    }],
                })
            session.committed_text = new_committed

        partial = text[len(session.committed_text):].strip()
        if partial:
            await safe_send(ws, {
                "type": "transcription",
                "results": [{
                    "is_final": False,
                    "alternatives": [{"transcript": partial, "words": []}],
                }],
            })

        session.last_full_text = text


async def safe_send(ws: WebSocket, msg: dict):
    try:
        await ws.send_text(json.dumps(msg))
    except Exception:
        pass


# -------------------- websocket endpoint --------------------

@app.websocket("/transcribe")
async def transcribe_ws(websocket: WebSocket):
    global active_sessions
    await websocket.accept()
    active_sessions += 1
    log.info(f"client connected (active={active_sessions})")

    session: Optional[StreamSession] = None
    task: Optional[asyncio.Task] = None

    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            mtype = msg.get("type")

            if mtype == "config":
                cfg = msg.get("config", {}) or {}
                lang_code = cfg.get("language", "en-US")
                lang = LANGUAGE_MAP.get(lang_code) or LANGUAGE_MAP.get(lang_code.split("-")[0].lower())
                if lang is None:
                    await safe_send(websocket, {"type": "error", "error": f"unsupported language: {lang_code}"})
                    await websocket.close()
                    return
                sr = int(cfg.get("sampleRate", SAMPLE_RATE_DEFAULT))
                session = StreamSession(language=lang, sample_rate=sr)
                task = asyncio.create_task(transcribe_loop(websocket, session))
                log.info(f"session start lang={lang} sr={sr}")
                await safe_send(websocket, {"type": "ready"})

            elif mtype == "audio":
                if session is None:
                    await safe_send(websocket, {"type": "error", "error": "audio before config"})
                    continue
                audio_b64 = msg.get("audio", "")
                if audio_b64:
                    session.append(base64.b64decode(audio_b64))

            elif mtype == "done":
                log.info("client done")
                break

            else:
                log.warning(f"unknown message type: {mtype}")

    except WebSocketDisconnect:
        log.info("client disconnected")
    except Exception:
        log.exception("websocket error")
    finally:
        if session:
            session.stop()
            try:
                final_text = await run_transcribe(session)
                remaining = final_text[len(session.committed_text):].strip()
                if remaining:
                    await safe_send(websocket, {
                        "type": "transcription",
                        "results": [{
                            "is_final": True,
                            "alternatives": [{"transcript": remaining, "words": []}],
                        }],
                    })
            except Exception:
                log.exception("final transcribe failed")

        if task:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

        await safe_send(websocket, {"type": "done"})
        try:
            await websocket.close()
        except Exception:
            pass

        active_sessions -= 1
        log.info(f"session end (active={active_sessions})")


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, ws_max_size=16 * 1024 * 1024, log_level=LOG_LEVEL.lower())