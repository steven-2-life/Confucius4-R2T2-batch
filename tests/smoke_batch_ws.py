"""Direct model WS smoke test. Prints hashes/counts, never transcript or key."""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import time
import uuid
import wave
from urllib.parse import urlsplit
from urllib.request import urlopen

import websockets


async def run(args, pcm):
    ready = 0
    barrier = asyncio.Event()
    observed_batches = {}
    async def stream(index):
        nonlocal ready
        texts, sizes, costs, queues, pending = [], [], [], [], []
        first_text = None
        processed = 0
        final = False
        sent_end = False
        eos_at = None
        processed_at_audio_end = 0
        modes, scopes = set(), set()
        waits = []
        configured_chunks, audio_chunks = set(), []
        async with websockets.connect(args.url, ping_interval=None, open_timeout=30, close_timeout=5) as ws:
            await ws.send(json.dumps(dict(channels=1, sample_rate=16000, requestId=uuid.uuid4().hex,
                secret_key=os.getenv('R2T2_SECRET_KEY', 'test0102'), language='Chinese', mode='slow', use_vad=False)))
            ack = json.loads(await asyncio.wait_for(ws.recv(), 30))
            if ack.get('status') != 'connected':
                raise RuntimeError('Model handshake failed')
            ready += 1
            if ready == args.streams:
                barrier.set()
            await asyncio.wait_for(barrier.wait(), 60)
            started = time.perf_counter()
            async def receive():
                nonlocal first_text, processed, final
                async for raw in ws:
                    data = json.loads(raw)
                    if data.get('status') == 'error':
                        raise RuntimeError(data.get('msg'))
                    msg = data.get('msg')
                    if not isinstance(msg, dict):
                        continue
                    text = msg.get('text', '')
                    if text:
                        if first_text is None:
                            first_text = (time.perf_counter() - started) * 1000
                        texts.append(text)
                    if 'batch_size' in msg:
                        if 'configured_chunk_ms' in msg:
                            configured_chunks.add(msg['configured_chunk_ms'])
                        if 'chunk_audio_ms' in msg:
                            audio_chunks.append(msg['chunk_audio_ms'])
                        sizes.append(msg['batch_size'])
                        costs.append(msg['batch_infer_ms'])
                        queues.append(msg['infer_queue_ms'])
                        pending.append(msg.get('pending_audio_ms', 0))
                        modes.add(msg.get('execution_mode', 'batch'))
                        scopes.add(msg.get('infer_ms_scope', 'shared_batch'))
                        waits.append(msg.get('batch_wait_ms', 0))
                        batch_id = msg.get('batch_total_calls')
                        if batch_id is not None:
                            observed_batches.setdefault(batch_id, dict(
                                batch_id=batch_id, batch_size=msg['batch_size'],
                                batch_infer_ms=msg['batch_infer_ms']))
                    processed = max(processed, msg.get('audio_processed_samples', 0))
                    final = bool(msg.get('reset'))
                if not sent_end:
                    raise RuntimeError('Model closed before audio completed')
            receiver = asyncio.create_task(receive())
            try:
                for start in range(0, len(pcm), 15360):
                    if args.realtime:
                        target = started + min(start + 15360, len(pcm)) / 32000
                        await asyncio.sleep(max(0, target - time.perf_counter()))
                    if receiver.done():
                        await receiver
                        raise RuntimeError('Receiver stopped before EOS')
                    await ws.send(pcm[start:start + 15360])
                sent_end = True
                processed_at_audio_end = processed
                eos_at = time.perf_counter()
                await ws.send('YOUDAO_ONETIME_ASR_STREAM_EOS')
                await asyncio.wait_for(receiver, args.timeout)
            finally:
                receiver.cancel()
                await asyncio.gather(receiver, return_exceptions=True)
            text = ''.join(texts)
            result = dict(stream=index, audio_seconds=len(pcm) / 32000,
                elapsed_seconds=time.perf_counter() - started, first_text_ms=first_text,
                text_chars=len(text), text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                processed_samples=processed, expected_samples=len(pcm) // 2,
                eos_completed=final and processed == len(pcm) // 2,
                batch_max=max(sizes, default=0), batch_mean=sum(sizes) / len(sizes) if sizes else 0,
                batch_calls=len(sizes), batch_infer_mean_ms=sum(costs) / len(costs) if costs else None,
                execution_modes=sorted(modes), infer_ms_scopes=sorted(scopes),
                configured_chunk_ms=sorted(configured_chunks),
                first_chunk_audio_ms=audio_chunks[0] if audio_chunks else None,
                subsequent_chunk_audio_ms=sorted(set(audio_chunks[1:])),
                batch_wait_max_ms=max(waits, default=0),
                batch_infer_p50_ms=sorted(costs)[len(costs)//2] if costs else None,
                batch_infer_p95_ms=sorted(costs)[min(len(costs)-1, int(len(costs)*.95))] if costs else None,
                pending_audio_max_ms=max(pending, default=0),
                processed_samples_at_audio_end=processed_at_audio_end,
                end_to_close_ms=(time.perf_counter() - eos_at) * 1000 if eos_at else None,
                scheduler_queue_max_ms=max(queues, default=0))
            return result
    results = await asyncio.gather(*(stream(i) for i in range(args.streams)))
    return results, sorted(observed_batches.values(), key=lambda x: x['batch_id'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('wav', type=Path)
    p.add_argument('--url', default='ws://127.0.0.1:8272/asr_stream_api_v1')
    p.add_argument('--streams', type=int, default=8)
    p.add_argument('--seconds', type=float, default=20)
    p.add_argument('--timeout', type=float, default=120)
    p.add_argument('--realtime', action='store_true')
    p.add_argument('--wait-ready', type=float, default=0, help='Wait for the new /health endpoint, seconds')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.wait_ready:
        origin = urlsplit(args.url)
        health_url = ('https' if origin.scheme == 'wss' else 'http') + '://' + origin.netloc + '/health'
        deadline = time.monotonic() + args.wait_ready
        while True:
            try:
                with urlopen(health_url, timeout=2) as response:
                    status = json.loads(response.read())
                    if status.get('status') == 'ok' and status.get('mode') in ('dynamic_batch', 'serial'):
                        break
            except Exception:
                pass
            if time.monotonic() >= deadline:
                raise TimeoutError('Batch model did not become healthy before deadline')
            time.sleep(2)
    with wave.open(str(args.wav), 'rb') as wav:
        if (wav.getnchannels(), wav.getframerate(), wav.getsampwidth()) != (1, 16000, 2):
            p.error('Use mono 16k PCM16 WAV')
        pcm = wav.readframes(round(args.seconds * 16000))
    results, batches = asyncio.run(run(args, pcm))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    durations = sorted(b['batch_infer_ms'] for b in batches)
    batch_summary = dict(unique_batches=len(batches),
        batch_infer_mean_ms=sum(durations) / len(durations) if durations else None,
        batch_infer_p50_ms=durations[len(durations)//2] if durations else None,
        batch_infer_p95_ms=durations[min(len(durations)-1, int(len(durations)*.95))] if durations else None,
        batch_size_mean=sum(b['batch_size'] for b in batches) / len(batches) if batches else None,
        batch_size_max=max((b['batch_size'] for b in batches), default=0), batches=batches)
    args.output.with_suffix('.batches.json').write_text(json.dumps(batch_summary, indent=2), encoding='utf-8')
    print(json.dumps(results, ensure_ascii=False, indent=2))
    if not all(r['eos_completed'] and r['text_chars'] for r in results):
        raise SystemExit(2)


if __name__ == '__main__':
    main()
