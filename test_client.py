"""Standalone smoke test. Streams a WAV file at real-time pace through the server."""

import argparse
import asyncio
import base64
import json
import time

import soundfile as sf
import websockets

import ssl
ssl_ctx = ssl.create_default_context()
ssl_ctx.check_hostname = False
ssl_ctx.verify_mode = ssl.CERT_NONE


async def stream_file(uri: str, wav_path: str, language: str, chunk_ms: int = 100):
    audio, sr = sf.read(wav_path, dtype="int16")
    if audio.ndim > 1:
        audio = audio[:, 0]
    chunk_samples = int(sr * chunk_ms / 1000)

    final_segments: list[str] = []

    ssl_arg = ssl_ctx if uri.startswith("wss://") else None
    async with websockets.connect(uri, max_size=16 * 1024 * 1024, ssl=ssl_arg) as ws:
        await ws.send(json.dumps({
            "type": "config",
            "config": {"language": language, "sampleRate": sr},
        }))

        async def reader():
            async for raw in ws:
                msg = json.loads(raw)
                if msg.get("type") == "transcription":
                    for r in msg["results"]:
                        tag = "FINAL" if r["is_final"] else "partial"
                        text = r["alternatives"][0]["transcript"]
                        print(f"[{tag}] {text}")
                        if r["is_final"] and text:
                            final_segments.append(text)
                elif msg.get("type") == "ready":
                    print("[ready]")
                elif msg.get("type") == "done":
                    print("[done]")
                    return
                elif msg.get("type") == "error":
                    print(f"[error] {msg['error']}")
                    return

        reader_task = asyncio.create_task(reader())

        for i in range(0, len(audio), chunk_samples):
            chunk = audio[i : i + chunk_samples].tobytes()
            await ws.send(json.dumps({
                "type": "audio",
                "audio": base64.b64encode(chunk).decode("ascii"),
            }))
            await asyncio.sleep(chunk_ms / 1000)

        await ws.send(json.dumps({"type": "done"}))
        await reader_task

    return final_segments


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--uri", default="ws://localhost:8765/transcribe")
    p.add_argument("--wav", required=True)
    p.add_argument("--language", default="de-DE")
    args = p.parse_args()

    t0 = time.time()
    final_segments = asyncio.run(stream_file(args.uri, args.wav, args.language))
    print(f"elapsed: {time.time() - t0:.1f}s")
    print("\n=== full transcript ===")
    print(" ".join(final_segments))