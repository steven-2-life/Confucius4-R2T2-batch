"""R2T2 v1 protocol with one isolated model owner and cross-call batching.

Uses ws_server.py's existing CLI, checkpoint loader, authentication and text
rules. Start with the SAME arguments as ws_server.py. Original server stays
available for rollback; model weights are loaded once, only in the worker.
"""
import asyncio
from collections import deque
import contextlib
import json
import os
import re
import time
import uuid
import wave

import numpy as np
from sanic import Sanic, json as json_response

import ws_server as legacy
from r2t2_batch import ModelProcess, parse_batch_enabled, configure_geometry


app = Sanic('r2t2_batch_server')
app.config.WEBSOCKET_PING_INTERVAL = None
app.config.WEBSOCKET_PING_TIMEOUT = None
RATE = 16000
configure_geometry(legacy)
EOS = legacy.YOUDAO_ONETIME_ASR_EOS_STRING
MAX_BATCH = int(os.getenv('R2T2_BATCH_MAX_SIZE', '8'))
WAIT_MS = float(os.getenv('R2T2_BATCH_WAIT_MS', '5'))
BATCH_ENABLED = parse_batch_enabled(os.getenv('R2T2_BATCH_ENABLED', '1'))
QUEUE_SECONDS = float(os.getenv('R2T2_BATCH_QUEUE_SECONDS', '30'))
if not 1 <= MAX_BATCH <= 64 or not 0 <= WAIT_MS <= 100 or not 1 <= QUEUE_SECONDS <= 300:
    raise ValueError('Invalid R2T2 batch/queue configuration')


@app.before_server_start
async def start_model(app):
    app.ctx.model = ModelProcess(MAX_BATCH, WAIT_MS, BATCH_ENABLED)
    app.ctx.sessions = set()
    await app.ctx.model.start()
    legacy.logger.info('R2T2 ready: batch_enabled=%s max_size=%s wait_ms=%s model_pid=%s',
                       BATCH_ENABLED, MAX_BATCH, WAIT_MS if BATCH_ENABLED else 0,
                       app.ctx.model.process.pid)


@app.after_server_stop
async def stop_model(app):
    await app.ctx.model.close()


@app.get('/health')
async def health(request):
    alive = request.app.ctx.model.process.is_alive()
    return json_response(dict(status='ok' if alive else 'error',
                              mode='dynamic_batch' if BATCH_ENABLED else 'serial',
                              batch_enabled=BATCH_ENABLED,
                              max_batch_size=MAX_BATCH if BATCH_ENABLED else 1,
                              max_wait_ms=WAIT_MS if BATCH_ENABLED else 0,
                              chunk_ms=legacy.STEP_MS, lookahead_ms=legacy.LOOKAHEAD_MS,
                              first_chunk_ms=legacy.STEP_MS + legacy.LOOKAHEAD_MS,
                              active_sessions=len(request.app.ctx.sessions)), status=200 if alive else 503)


class ProtocolSession:
    def __init__(self, client, header, send):
        self.client, self.header, self.send = client, header, send
        # Internal handle is never reused even when an external requestId repeats.
        self.key = uuid.uuid4().hex
        self.request_id = header['requestId']
        self.use_vad = bool(header.get('use_vad', False))
        self.rollback = header.get('mode', 'slow') != 'slow'
        self.buffer = np.empty(0, dtype=np.float32)
        self.first = True
        self.processed = 0
        self.received_samples = 0
        self.last_fixed = ''
        self.tokens = deque(maxlen=legacy.MAX_TOKENS)
        self.max_tokens = max(1, int((legacy.STEP_MS + legacy.LOOKAHEAD_MS) / 80))
        self.latest_text = ''
        self.hallucination = False
        self.metrics = {}

    async def open(self):
        language = self.header.get('language', 'zhen')
        await self.client.call(self.key, 'open', vad_enabled=self.use_vad, options=dict(
            context=legacy.resolve_qwen_context(self.header, self.header.get('smooth', False)),
            language=None if language == 'zhen' else language,
            unfixed_chunk_num=0, unfixed_token_num=legacy.UNFIX_TOKEN_NUM,
            chunk_size_sec=legacy.CHUNK_ASR_SECONDS))

    async def emit(self, msg):
        await self.send(json.dumps(dict(status='success', requestId=self.request_id, msg=msg), ensure_ascii=False))

    async def reset(self):
        await self.client.call(self.key, 'reset')
        self.last_fixed = ''
        self.latest_text = ''
        self.hallucination = False

    async def _infer(self, audio, chunk_samples, received_at):
        began = time.perf_counter()
        result = await self.client.call(self.key, 'infer', audio=audio.astype('<f4', copy=False).tobytes(),
                                        chunk_samples=chunk_samples, max_tokens=int(self.max_tokens),
                                        rollback=self.rollback)
        self.metrics = result
        self.processed += len(audio)
        self.latest_text = result['text']
        fixed = result['fixed_text']
        delta = fixed[len(self.last_fixed):] if len(fixed) > len(self.last_fixed) else ''
        if delta:
            self.last_fixed = fixed
            self.tokens.extend(legacy.split_text_to_tokens(delta))
            self.max_tokens = max(1, int(legacy.STEP_MS / 80))
        elif not legacy.is_last_token_chinese(list(self.tokens)):
            self.max_tokens += .5
        else:
            self.max_tokens = max(1, int(legacy.STEP_MS / 80))
        if legacy.is_last_token_chinese(list(self.tokens)):
            self.max_tokens *= 2
        self.max_tokens = min(min(32, max(4, 2 * int(legacy.STEP_MS / 80))), self.max_tokens)
        hallucination, _ = legacy.detect_hallucination(self.request_id, self.last_fixed)
        self.hallucination = self.hallucination or hallucination
        reset = bool(result['vad_end'] and self.use_vad)
        if reset:
            delta = self.latest_text[len(self.last_fixed):] if len(self.latest_text) > len(self.last_fixed) else ''
            await self.reset()
        cost = result['batch_infer_ms']
        msg = dict(text=delta, reset=reset, asr_cost_ms=cost,
                   chunk_audio_ms=len(audio) / 16, configured_chunk_ms=legacy.STEP_MS,
                   total_cost_ms=cost + result['vad_cost_ms'],
                   lag_ms=(time.perf_counter() - received_at) * 1000,
                   audio_processed_samples=self.processed,
                   infer_queue_ms=result['scheduler_queue_ms'],
                   infer_ms_scope=result.get('infer_ms_scope', 'shared_batch'),
                   execution_mode=result.get('execution_mode', 'batch'), batch_size=result['batch_size'],
                   batch_infer_ms=cost, batch_wait_ms=result['batch_wait_ms'],
                   request_service_ms=(time.perf_counter() - began) * 1000,
                   receive_to_step_start_ms=(began - received_at) * 1000,
                   pending_audio_ms=max(0, self.received_samples - self.processed) / 16,
                   processed_audio_ms=self.processed / 16,
                   batch_total_items=result['batch_total_items'],
                   batch_total_calls=result['batch_total_calls'],
                   max_observed_batch=result['max_observed_batch'])
        await self.emit(msg)

    async def feed(self, audio, received_at):
        self.buffer = np.concatenate((self.buffer, audio))
        while True:
            size = int((legacy.STEP_MS + legacy.LOOKAHEAD_MS) * RATE / 1000) if self.first else legacy.CHUNK_ASR_SIZE
            if len(self.buffer) < size:
                break
            chunk = self.buffer[:size]
            self.buffer = self.buffer[size:]
            self.first = False
            await self._infer(chunk, size, received_at)
        if self.hallucination:
            result = await self.client.call(self.key, 'finish', max_tokens=max(1, int(
                (legacy.STEP_MS + legacy.LOOKAHEAD_MS) / 80)))
            fixed = result['fixed_text'].split('|')[0]
            await self.emit(dict(text=fixed[len(self.last_fixed):] if len(fixed) > len(self.last_fixed) else '', reset=True))
            await self.reset()
        await self.send('{}')

    async def finish(self):
        if len(self.buffer):
            # Preserve a short final packet in the model's tail buffer. Do not
            # force it into a full normal step or silently discard it.
            size = int((legacy.STEP_MS + legacy.LOOKAHEAD_MS) * RATE / 1000) if self.first else legacy.CHUNK_ASR_SIZE
            await self.client.call(self.key, 'infer', audio=self.buffer.astype('<f4').tobytes(),
                                   chunk_samples=size, max_tokens=int(self.max_tokens), rollback=self.rollback)
            self.processed += len(self.buffer)
            self.buffer = np.empty(0, dtype=np.float32)
        result = await self.client.call(self.key, 'finish', max_tokens=max(1, int(
            (legacy.STEP_MS + legacy.LOOKAHEAD_MS) / 80)))
        fixed = result['fixed_text'].split('|')[0]
        await self.emit(dict(text=fixed[len(self.last_fixed):] if len(fixed) > len(self.last_fixed) else '',
                             reset=True, audio_processed_samples=self.processed))


@app.websocket('/asr_stream_api_v1')
async def stream(request, ws):
    session = None
    receiver = None
    writer = None
    request_id = None
    pending_samples = 0
    queue = asyncio.Queue(maxsize=256)
    try:
        raw = await asyncio.wait_for(ws.recv(), 10)
        header = json.loads(raw)
        request_id = header.get('requestId')
        if not isinstance(request_id, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,128}', request_id):
            raise ValueError('requestId must be 1-128 letters, digits, hyphens or underscores')
        if header.get('secret_key') not in legacy.secret_key_list:
            await ws.close(code=4401, reason='Unauthorized')
            return
        if header.get('channels', 1) != 1 or header.get('sample_rate', RATE) != RATE:
            raise ValueError('Only mono 16000 Hz PCM16 is supported')
        if request_id in request.app.ctx.sessions:
            raise ValueError('requestId is already active')
        if len(request.app.ctx.sessions) >= 16:
            raise ValueError('Maximum active sessions reached')
        legacy.resolve_system_prompt(header)
        session = ProtocolSession(request.app.ctx.model, header, ws.send)
        await session.open()
        request.app.ctx.sessions.add(request_id)
        await ws.send(json.dumps(dict(status='connected', requestId=request_id, msg='',
                                       active_connections=len(request.app.ctx.sessions))))
        if legacy.args.save_audio:
            path = os.path.join(legacy.WAV_STORE_DIR, request_id + '.wav')
            writer = wave.open(path, 'wb')
            writer.setparams((1, 2, RATE, 0, 'NONE', 'not compressed'))

        async def receive():
            nonlocal pending_samples
            first_packet = True
            try:
                while True:
                    raw = await asyncio.wait_for(ws.recv(), 120)
                    received_at = time.perf_counter()
                    if isinstance(raw, str):
                        if raw == EOS:
                            await queue.put(('end', None, received_at))
                            return
                        continue
                    if not isinstance(raw, bytes) or not raw or len(raw) % 2:
                        raise ValueError('Expected non-empty PCM16 binary frame')
                    if first_packet and legacy.if_contains_wav_header(raw):
                        raw = raw[44:]
                    first_packet = False
                    pending_samples += len(raw) // 2
                    if pending_samples > QUEUE_SECONDS * RATE:
                        raise ValueError('Audio queue overflow; no audio was skipped or replayed')
                    if writer:
                        writer.writeframes(raw)
                    audio = np.frombuffer(raw, dtype='<i2').astype(np.float32) / 32768
                    session.received_samples += len(audio)
                    await queue.put(('audio', audio, received_at))
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # Don't wait behind abandoned audio after overload/disconnect.
                while not queue.empty():
                    queue.get_nowait()
                queue.put_nowait(('error', exc, time.perf_counter()))
                with contextlib.suppress(Exception):
                    await ws.close(code=1013, reason='R2T2 input queue or connection failed')

        receiver = asyncio.create_task(receive())
        while True:
            kind, value, received_at = await queue.get()
            if kind == 'error':
                raise value
            if kind == 'end':
                await session.finish()
                break
            before = session.processed
            await session.feed(value, received_at)
            pending_samples -= session.processed - before
    except Exception as exc:
        legacy.logger.exception('R2T2 batch session failed requestId=%s', request_id)
        with contextlib.suppress(Exception):
            await ws.send(json.dumps(dict(status='error', requestId=request_id,
                                           msg=f'{type(exc).__name__}: {exc}')))
    finally:
        if receiver:
            receiver.cancel()
            await asyncio.gather(receiver, return_exceptions=True)
        if session:
            with contextlib.suppress(Exception):
                await request.app.ctx.model.call(session.key, 'close')
            request.app.ctx.sessions.discard(request_id)
        if writer:
            writer.close()
        with contextlib.suppress(Exception):
            await ws.close()


if __name__ == '__main__':
    app.run(host=legacy.args.ip, port=int(legacy.args.port), single_process=True)
