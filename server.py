"""Production async streaming server for Qwen3-ASR using vLLM AsyncLLMEngine.

Architecture:
  - One shared AsyncLLMEngine handles all concurrent requests via continuous batching
  - One shared Qwen3ASRModel helper (CPU only) for prompt building + output parsing
  - Per-WebSocket: a StreamSession with its own audio buffer, prefix tracking, request lifecycle
  - Engine batches across all sessions automatically — no global lock, no thread pool

Protocol (unchanged from previous version):
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
from typing import Optional

import torch
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from qwen_asr import Qwen3ASRModel
from vllm import AsyncEngineArgs, AsyncLLMEngine

from asr_helpers import LANGUAGE_MAP, resolve_language
from session import StreamSession

# -------------------- config --------------------

MODEL_PATH = os.environ.get("MODEL_PATH", "Qwen/Qwen3-ASR-0.6B")
PORT = int(os.environ.get("PORT", "8765"))
HOST = os.environ.get("HOST", "0.0.0.0")
GPU_MEMORY_UTILIZATION = float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.85"))
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "8192"))
MAX_NUM_SEQS = int(os.environ.get("MAX_NUM_SEQS", "256"))
DTYPE = os.environ.get("DTYPE", "bfloat16")
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
log = logging.getLogger("server")

# globals (initialized at startup)
engine: Optional[AsyncLLMEngine] = None
helper: Optional[Qwen3ASRModel] = None

# metrics
active_sessions = 0
total_sessions = 0

app = FastAPI()


# -------------------- lifecycle --------------------

@app.on_event("startup")
async def startup():
    global engine, helper

    log.info(f"Loading prompt-builder helper on CPU ({MODEL_PATH})...")
    # CPU-only helper — gives us _build_text_prompt() and processor/tokenizer for prefix rollback.
    # No GPU memory cost. ~1.5GB CPU RAM.
    helper = Qwen3ASRModel.from_pretrained(
        MODEL_PATH,
        dtype=torch.float32,
        device_map="cpu",
    )
    log.info("Helper ready (CPU).")

    log.info(f"Initializing AsyncLLMEngine for {MODEL_PATH} on GPU...")
    engine_args = AsyncEngineArgs(
        model=MODEL_PATH,
        dtype=DTYPE,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=MAX_NUM_SEQS,
        trust_remote_code=True,
        enforce_eager=False,
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    log.info(f"AsyncLLMEngine ready (max_num_seqs={MAX_NUM_SEQS}, gpu_mem={GPU_MEMORY_UTILIZATION}).")


@app.on_event("shutdown")
async def shutdown():
    log.info(f"Shutdown initiated. active_sessions={active_sessions}")


# -------------------- health --------------------

@app.get("/health")
async def health():
    return {
        "status": "ok" if engine is not None else "loading",
        "model": MODEL_PATH,
        "engine_ready": engine is not None,
        "active_sessions": active_sessions,
        "total_sessions_served": total_sessions,
    }


# -------------------- websocket --------------------

@app.websocket("/transcribe")
async def transcribe_ws(websocket: WebSocket):
    global active_sessions, total_sessions

    await websocket.accept()
    total_sessions += 1
    sid = total_sessions
    active_sessions += 1
    log.info(f"[{sid}] connected (active={active_sessions})")

    session: Optional[StreamSession] = None

    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            mtype = msg.get("type")

            if mtype == "config":
                if session is not None:
                    await _send(websocket, {"type": "error", "error": "config already received"})
                    continue

                cfg = msg.get("config", {}) or {}
                lang_code = cfg.get("language", "en-US")
                lang = resolve_language(lang_code)
                if lang is None:
                    await _send(websocket, {"type": "error", "error": f"unsupported language: {lang_code}"})
                    await websocket.close()
                    return

                sr = int(cfg.get("sampleRate", 16000))

                session = StreamSession(
                    session_id=sid,
                    websocket=websocket,
                    engine=engine,
                    helper=helper,
                    force_language=lang,
                    sample_rate=sr,
                )
                await session.start()
                log.info(f"[{sid}] session started lang={lang} sr={sr}")
                await _send(websocket, {"type": "ready"})

            elif mtype == "audio":
                if session is None:
                    await _send(websocket, {"type": "error", "error": "audio before config"})
                    continue
                audio_b64 = msg.get("audio")
                if audio_b64:
                    session.append_audio(base64.b64decode(audio_b64))

            elif mtype == "done":
                log.info(f"[{sid}] client signalled done")
                break

            else:
                log.warning(f"[{sid}] unknown message type: {mtype}")

    except WebSocketDisconnect:
        log.info(f"[{sid}] client disconnected")
    except Exception:
        log.exception(f"[{sid}] websocket loop error")
    finally:
        if session:
            try:
                await session.finalize()
            except Exception:
                log.exception(f"[{sid}] finalize error")
            await session.close()

        await _send(websocket, {"type": "done"})
        try:
            await websocket.close()
        except Exception:
            pass

        active_sessions -= 1
        log.info(f"[{sid}] cleanup done (active={active_sessions})")


async def _send(ws: WebSocket, msg: dict):
    try:
        await ws.send_text(json.dumps(msg))
    except Exception:
        pass


# -------------------- main --------------------

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        ws_max_size=16 * 1024 * 1024,
        log_level=LOG_LEVEL.lower(),
    )