"""Qwen3-ASR WebSocket streaming server (vLLM backend, native streaming)."""

import asyncio
import base64
import json
import logging
import os
import time
from typing import Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from qwen_asr import Qwen3ASRModel

# -------------------- config --------------------

MODEL_PATH = os.environ.get("MODEL_PATH", "Qwen/Qwen3-ASR-0.6B")
PORT = int(os.environ.get("PORT", "8765"))
HOST = os.environ.get("HOST", "0.0.0.0")
SAMPLE_RATE_DEFAULT = 16000
GPU_MEMORY_UTILIZATION = float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.8"))
MAX_INFERENCE_BATCH_SIZE = int(os.environ.get("MAX_INFERENCE_BATCH_SIZE", "128"))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "32"))   # small for streaming
STEP_MS = int(os.environ.get("STEP_MS", "500"))                # how often we feed audio to the model
UNFIXED_CHUNK_NUM = int(os.environ.get("UNFIXED_CHUNK_NUM", "4"))
UNFIXED_TOKEN_NUM = int(os.environ.get("UNFIXED_TOKEN_NUM", "5"))
CHUNK_SIZE_SEC = float(os.environ.get("CHUNK_SIZE_SEC", "1.0"))
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")

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
    log.info(f"Loading {MODEL_PATH} (vLLM backend)...")
    t0 = time.time()
    model = Qwen3ASRModel.LLM(
        model=MODEL_PATH,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        max_inference_batch_size=MAX_INFERENCE_BATCH_SIZE,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    log.info(f"Model loaded in {time.time() - t0:.1f}s")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": MODEL_PATH,
        "loaded": model is not None,
        "active_sessions": active_sessions,
        "backend": "vllm",
    }


# -------------------- per-session state --------------------

class StreamSession:
    def __init__(self, language: str, sample_rate: int):
        self.language = language
        self.sample_rate = sample_rate
        self.audio_buffer = bytearray()
        self.last_full_text = ""
        self.committed_text = ""
        self.state = None
        self.running = True

    def init_state(self):
        self.state = model.init_streaming_state(
            unfixed_chunk_num=UNFIXED_CHUNK_NUM,
            unfixed_token_num=UNFIXED_TOKEN_NUM,
            chunk_size_sec=CHUNK_SIZE_SEC,
        )
        self.state.force_language = self.language

    def append(self, pcm: bytes):
        self.audio_buffer.extend(pcm)

    def pop_step(self) -> Optional[np.ndarray]:
        """Pop STEP_MS worth of audio. Returns None if not enough buffered."""
        step_bytes = int(STEP_MS / 1000 * self.sample_rate * 2)
        if len(self.audio_buffer) < step_bytes:
            return None
        chunk = bytes(self.audio_buffer[:step_bytes])
        del self.audio_buffer[:step_bytes]
        return np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0

    def drain(self) -> Optional[np.ndarray]:
        """Pop everything left in buffer."""
        if not self.audio_buffer:
            return None
        chunk = bytes(self.audio_buffer)
        self.audio_buffer.clear()
        return np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0

    def stop(self):
        self.running = False


def stable_prefix(prev: str, curr: str, after: str) -> str:
    """LocalAgreement-2 word-level common prefix."""
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

async def feed_chunk(session: StreamSession, segment: np.ndarray):
    """Feed one segment to the model and update state.text."""
    def _call():
        model.streaming_transcribe(segment, session.state)
        return session.state.text or ""
    return await asyncio.to_thread(_call)


async def transcribe_loop(ws: WebSocket, session: StreamSession):
    poll_interval = STEP_MS / 1000 / 2  # poll twice per step
    while session.running:
        await asyncio.sleep(poll_interval)
        if not session.running:
            break

        segment = session.pop_step()
        if segment is None:
            continue

        try:
            text = await feed_chunk(session, segment)
        except Exception as e:
            log.exception("streaming_transcribe failed")
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
                # init_streaming_state runs sync, may take a few hundred ms
                await asyncio.to_thread(session.init_state)
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
        if session and session.state is not None:
            session.stop()

            # flush any remaining audio
            tail = session.drain()
            if tail is not None and len(tail) > 0:
                try:
                    await feed_chunk(session, tail)
                except Exception:
                    log.exception("tail feed failed")

            # finalize
            try:
                def _finish():
                    model.finish_streaming_transcribe(session.state)
                    return session.state.text or ""
                final_text = await asyncio.to_thread(_finish)
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
                log.exception("finish_streaming_transcribe failed")

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