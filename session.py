"""Per-WebSocket session: audio buffering, streaming inference, prefix-rollback logic."""

import asyncio
import json
import logging
from typing import Optional

import numpy as np
from fastapi import WebSocket
from qwen_asr import Qwen3ASRModel
from qwen_asr.inference.qwen3_asr import parse_asr_output
from vllm import AsyncLLMEngine
from vllm.sampling_params import SamplingParams

log = logging.getLogger("session")

# Tunables (per-session; could be overridden by client config later)
CHUNK_SIZE_SEC = 1.0
UNFIXED_CHUNK_NUM = 4
UNFIXED_TOKEN_NUM = 5
MAX_NEW_TOKENS_PER_CHUNK = 32
MAX_AUDIO_ACCUM_SEC = 60          # safety cap; older audio dropped past this
WAIT_TIMEOUT_SEC = 30.0           # how long to wait for new audio before checking running flag


class StreamSession:
    def __init__(
        self,
        session_id: int,
        websocket: WebSocket,
        engine: AsyncLLMEngine,
        helper: Qwen3ASRModel,
        force_language: str,
        sample_rate: int = 16000,
    ):
        self.sid = session_id
        self.ws = websocket
        self.engine = engine
        self.helper = helper
        self.force_language = force_language
        self.sample_rate = sample_rate
        self.tokenizer = helper.processor.tokenizer

        # Audio buffers (in float32, [-1, 1])
        self._incoming = np.zeros(0, dtype=np.float32)   # raw incoming, not yet chunked
        self._accum = np.zeros(0, dtype=np.float32)      # all accumulated audio for re-feed
        self._chunk_size_samples = int(CHUNK_SIZE_SEC * sample_rate)
        self._max_accum_samples = MAX_AUDIO_ACCUM_SEC * sample_rate

        # Streaming state
        self._chunk_id = 0
        self._raw_decoded = ""           # accumulated raw model output (with format markers)
        self._committed_text = ""    # text already emitted as FINAL
        self._last_partial = ""

        # Build base prompt with language directive baked in
        self._prompt_raw = helper._build_text_prompt(
            context="",
            force_language=force_language,
        )

        # Lifecycle
        self._running = True
        self._audio_event = asyncio.Event()
        self._proc_task: Optional[asyncio.Task] = None
        self._current_request_id: Optional[str] = None
        self._lock = asyncio.Lock()  # serialize chunk processing within this session

    # ---------- public API (called from WS handler) ----------

    async def start(self):
        self._proc_task = asyncio.create_task(self._process_loop())

    def append_audio(self, pcm: bytes):
        """Receive raw int16 PCM bytes from the client, append to incoming buffer."""
        if not pcm:
            return
        chunk = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        self._incoming = np.concatenate([self._incoming, chunk])
        self._audio_event.set()

    async def finalize(self):
        async with self._lock:
            if len(self._incoming) > 0:
                self._accum = np.concatenate([self._accum, self._incoming])
                self._incoming = np.zeros(0, dtype=np.float32)
                self._cap_accum()
                if len(self._accum) >= int(0.3 * self.sample_rate):
                    await self._process_chunk_locked()

        # Anything still in the unstable tail becomes final at end-of-stream
        try:
            _, current_text = parse_asr_output(self._raw_decoded, user_language=self.force_language)
            remaining = current_text[len(self._committed_text):]
            if remaining.strip():
                await self._emit_final(remaining)
                self._committed_text = current_text
        except Exception:
            log.exception(f"[{self.sid}] finalize parse error")

    async def close(self):
        """Stop processing, abort in-flight vLLM request, await background task."""
        self._running = False
        self._audio_event.set()  # wake the loop

        # Abort any in-flight request to free a vLLM slot immediately
        if self._current_request_id:
            try:
                await self.engine.abort(self._current_request_id)
            except Exception:
                pass

        if self._proc_task and not self._proc_task.done():
            self._proc_task.cancel()
            try:
                await self._proc_task
            except (asyncio.CancelledError, Exception):
                pass

    # ---------- internal ----------

    async def _process_loop(self):
        """Run as background task. Waits for audio, processes one chunk at a time."""
        try:
            while self._running:
                if len(self._incoming) < self._chunk_size_samples:
                    self._audio_event.clear()
                    try:
                        await asyncio.wait_for(self._audio_event.wait(), timeout=WAIT_TIMEOUT_SEC)
                    except asyncio.TimeoutError:
                        continue
                    if not self._running:
                        return

                # Move one chunk's worth from incoming to accum
                async with self._lock:
                    while self._running and len(self._incoming) >= self._chunk_size_samples:
                        chunk = self._incoming[: self._chunk_size_samples]
                        self._incoming = self._incoming[self._chunk_size_samples:]
                        self._accum = np.concatenate([self._accum, chunk])
                        self._cap_accum()
                        await self._process_chunk_locked()

        except asyncio.CancelledError:
            return
        except Exception:
            log.exception(f"[{self.sid}] process_loop fatal")
            await self._send({"type": "error", "error": "internal error"})

    def _cap_accum(self):
        if len(self._accum) > self._max_accum_samples:
            drop = len(self._accum) - self._max_accum_samples
            self._accum = self._accum[drop:]
            log.debug(f"[{self.sid}] capped accum, dropped {drop} samples")

    async def _process_chunk_locked(self):
        """Run one inference step on current accumulated audio. Caller must hold self._lock."""
        prefix = self._build_prefix()
        prompt = self._prompt_raw + prefix

        sampling = SamplingParams(
            temperature=0.0,
            max_tokens=MAX_NEW_TOKENS_PER_CHUNK,
            skip_special_tokens=False,
        )

        request_id = f"sid{self.sid}-c{self._chunk_id}"
        self._current_request_id = request_id

        prompt_input = {
            "prompt": prompt,
            "multi_modal_data": {"audio": [self._accum]},
        }

        gen_text = ""
        try:
            async for output in self.engine.generate(
                prompt=prompt_input,
                sampling_params=sampling,
                request_id=request_id,
            ):
                if not self._running:
                    break
                gen_text = output.outputs[0].text
                if output.finished:
                    break
        except asyncio.CancelledError:
            try:
                await self.engine.abort(request_id)
            except Exception:
                pass
            raise
        except Exception as e:
            log.exception(f"[{self.sid}] inference error chunk={self._chunk_id}")
            await self._send({"type": "error", "error": f"inference: {e}"})
            self._current_request_id = None
            return
        finally:
            self._current_request_id = None

        # Update raw decoded (with prefix + new generation)
        # Update raw decoded
        self._raw_decoded = prefix + gen_text

        # Compute the "stable" portion: raw_decoded minus the last UNFIXED_TOKEN_NUM tokens
        # (these will be regenerated next chunk, so they're not committed).
        stable_raw = ""
        if self._chunk_id >= UNFIXED_CHUNK_NUM:
            ids = self.tokenizer.encode(self._raw_decoded)
            k = UNFIXED_TOKEN_NUM
            while True:
                end_idx = max(0, len(ids) - k)
                candidate = self.tokenizer.decode(ids[:end_idx]) if end_idx > 0 else ""
                if "\ufffd" not in candidate:
                    stable_raw = candidate
                    break
                if end_idx == 0:
                    stable_raw = ""
                    break
                k += 1

        # Parse both into clean text (stripping the "language X<asr_text>" markers)
        try:
            _, current_text = parse_asr_output(self._raw_decoded, user_language=self.force_language)
            if stable_raw:
                _, stable_text = parse_asr_output(stable_raw, user_language=self.force_language)
            else:
                stable_text = ""
        except Exception:
            log.exception(f"[{self.sid}] parse error")
            current_text = ""
            stable_text = ""

        # Emit any newly-finalized text
        if len(stable_text) > len(self._committed_text):
            new_final = stable_text[len(self._committed_text):]
            if new_final.strip():
                await self._emit_final(new_final)
            self._committed_text = stable_text

        # Emit the unstable tail as partial
        unstable = current_text[len(self._committed_text):]
        if unstable.strip() and unstable != self._last_partial:
            await self._emit_partial(unstable)
            self._last_partial = unstable

        self._chunk_id += 1

    def _build_prefix(self) -> str:
        """Replicate qwen-asr SDK's prefix-rollback logic for streaming continuity."""
        if self._chunk_id < UNFIXED_CHUNK_NUM:
            return ""
        cur_ids = self.tokenizer.encode(self._raw_decoded)
        k = UNFIXED_TOKEN_NUM
        while True:
            end_idx = max(0, len(cur_ids) - k)
            prefix = self.tokenizer.decode(cur_ids[:end_idx]) if end_idx > 0 else ""
            if "\ufffd" not in prefix:
                return prefix
            if end_idx == 0:
                return ""
            k += 1

    # ---------- output emission ----------

    async def _emit_partial(self, text: str):
        await self._send({
            "type": "transcription",
            "results": [{
                "is_final": False,
                "alternatives": [{"transcript": text, "words": []}],
            }],
        })

    async def _emit_final(self, text: str):
        await self._send({
            "type": "transcription",
            "results": [{
                "is_final": True,
                "alternatives": [{"transcript": text, "words": []}],
            }],
        })

    async def _send(self, msg: dict):
        try:
            await self.ws.send_text(json.dumps(msg))
        except Exception:
            self._running = False