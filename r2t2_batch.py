"""Single-owner model process and fair cross-session dynamic batching.

This module is importable without torch/vLLM for scheduler regression tests.
The network process never initializes CUDA or calls the inference engine.
"""
import asyncio
import contextlib
import copy
from dataclasses import dataclass, field
import logging
import multiprocessing as mp
import os
import queue
import time
import traceback
import uuid


def parse_batch_enabled(value):
    value = str(value).strip().lower()
    if value not in {'0', '1', 'false', 'true'}:
        raise ValueError('R2T2_BATCH_ENABLED must be 0/1 or false/true')
    return value in {'1', 'true'}


def configure_geometry(legacy):
    """Apply the same startup geometry in the network and model processes."""
    step = int(os.getenv('R2T2_CHUNK_MS', str(getattr(legacy, 'STEP_MS', 160))))
    lookahead = int(os.getenv('R2T2_LOOKAHEAD_MS', str(getattr(legacy, 'LOOKAHEAD_MS', 160))))
    if not 80 <= step <= 2000 or step % 80 or not 0 <= lookahead <= 2000 or lookahead % 80:
        raise ValueError('R2T2 chunk/lookahead must be multiples of 80ms; chunk 80-2000, lookahead 0-2000')
    legacy.STEP_MS = step
    legacy.LOOKAHEAD_MS = lookahead
    legacy.CHUNK_ASR_SECONDS = step / 1000
    legacy.CHUNK_ASR_SIZE = step * 16
    return step, lookahead


@dataclass
class Step:
    key: str
    generator: object
    future: object
    queued_at: float
    item: object = None
    params: object = None
    selected_at: float = 0.
    metrics: dict = field(default_factory=lambda: dict(batch_infer_ms=0., batch_size=0,
                                                      batch_wait_ms=0., scheduler_queue_ms=0.))


class BatchExecutor:
    def __init__(self, generate, max_size=8, wait_ms=5):
        if not 1 <= max_size <= 64 or not 0 <= wait_ms <= 100:
            raise ValueError('Invalid batch size/wait')
        self.generate, self.max_size, self.wait_ms = generate, max_size, wait_ms
        self.ready = asyncio.Queue()
        self.busy = set()
        self.batches = self.items = 0
        self.max_observed_batch = 0

    async def execute(self, key, generator):
        if key in self.busy:
            raise RuntimeError('Concurrent mutation of one session rejected')
        self.busy.add(key)
        future = asyncio.get_running_loop().create_future()
        job = Step(key, generator, future, time.perf_counter())
        try:
            try:
                job.item, job.params = next(generator)
            except StopIteration as done:
                return done.value, job.metrics
            await self.ready.put(job)
            return await future
        finally:
            self.busy.discard(key)
            generator.close()

    async def run(self):
        while True:
            first = await self.ready.get()
            first.selected_at = time.perf_counter()
            batch = [first]
            # Never extend the collection deadline after another arrival.
            deadline = asyncio.get_running_loop().time() + self.wait_ms / 1000
            while len(batch) < self.max_size:
                try:
                    job = self.ready.get_nowait()
                except asyncio.QueueEmpty:
                    remaining = deadline - asyncio.get_running_loop().time()
                    if remaining <= 0:
                        break
                    try:
                        job = await asyncio.wait_for(self.ready.get(), remaining)
                    except asyncio.TimeoutError:
                        break
                job.selected_at = time.perf_counter()
                batch.append(job)
            batch = [job for job in batch if not job.future.cancelled()]
            if not batch:
                continue
            started = time.perf_counter()
            try:
                if len({job.key for job in batch}) != len(batch):
                    raise RuntimeError('Duplicate session in batch')
                outputs = self.generate([j.item for j in batch], [j.params for j in batch])
                if len(outputs) != len(batch):
                    raise RuntimeError('Engine output count differs from batch size')
                elapsed = (time.perf_counter() - started) * 1000
                self.batches += 1
                self.items += len(batch)
                self.max_observed_batch = max(self.max_observed_batch, len(batch))
                for job, output in zip(batch, outputs):
                    if job.future.cancelled():
                        continue
                    job.metrics['batch_size'] = max(job.metrics['batch_size'], len(batch))
                    job.metrics['batch_infer_ms'] += elapsed
                    wait = (started - job.queued_at) * 1000
                    job.metrics['scheduler_queue_ms'] += wait
                    job.metrics['batch_wait_ms'] += (started - job.selected_at) * 1000
                    try:
                        job.item, job.params = job.generator.send([output])
                    except StopIteration as done:
                        job.future.set_result((done.value, job.metrics))
                    except Exception as exc:
                        job.future.set_exception(exc)
                    else:
                        # One decode from this session per batch. Rejoin at the end.
                        job.queued_at = time.perf_counter()
                        self.ready.put_nowait(job)
            except Exception as exc:
                for job in batch:
                    if not job.future.done():
                        job.future.set_exception(exc)


def clone_vad(template):
    """Share read-only weights; isolate feature, RNN and postprocessor caches."""
    from fireredvad import FireRedStreamVad
    vad = FireRedStreamVad(copy.deepcopy(template.audio_feat), template.vad_model,
                          copy.deepcopy(template.postprocessor), template.config)
    vad.reset()
    return vad


async def _worker_loop(incoming, outgoing, max_size, wait_ms, batch_enabled=True):
    # Import and initialize the existing loader here ONLY, preserving deployment
    # attention backend, memory budget, checkpoint and warm-up settings.
    import ws_server as legacy
    import numpy as np
    configure_geometry(legacy)
    await legacy.initialize_models(None)
    model = legacy.asr_model
    executor = BatchExecutor(lambda inputs, params: model.model.generate(
        inputs, sampling_params=params, use_tqdm=False), max_size, wait_ms)
    sessions, tasks = {}, {}
    runner = asyncio.create_task(executor.run()) if batch_enabled else None
    outgoing.put(dict(id='startup', value=dict(ready=True, batch_enabled=batch_enabled)))

    async def dispatch(command):
        request_id, key, op = command['id'], command['key'], command['op']
        try:
            if op == 'open':
                if key in sessions:
                    raise ValueError('Session already exists')
                if len(sessions) >= 16:
                    raise ValueError('Maximum active sessions reached')
                options = command['options']
                sessions[key] = dict(options=options, state=model.init_streaming_state(**options),
                                     vad=clone_vad(legacy.stream_vad) if command.get('vad_enabled', False) else None)
                value = dict(opened=True)
            elif op == 'close':
                task = tasks.get(key)
                if task and task is not asyncio.current_task():
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
                sessions.pop(key, None)
                value = dict(closed=True)
            else:
                session = sessions[key]
                state = session['state']
                if op == 'reset':
                    session['state'] = model.init_streaming_state(**session['options'])
                    if session['vad'] is not None:
                        session['vad'].reset()
                    value = dict(reset=True)
                else:
                    vad_ms, vad_end = 0., False
                    if op == 'infer':
                        audio = np.frombuffer(command['audio'], dtype='<f4')
                        state.chunk_size_samples = command['chunk_samples']
                        state.chunk_size_sec = command['chunk_samples'] / 16000
                        began = time.perf_counter()
                        events = session['vad'].detect_chunk(
                            np.clip(audio * 32768, -32768, 32767).astype(np.int16)) if session['vad'] is not None and len(audio) >= command['chunk_samples'] else []
                        vad_ms = (time.perf_counter() - began) * 1000
                        vad_end = any(e.is_speech_end for e in events)
                    elif op == 'finish':
                        pass
                    else:
                        raise ValueError('Unknown worker operation')
                    if batch_enabled:
                        generator = (model._streaming_transcribe_steps(
                            audio, state, command['max_tokens'], command['rollback']) if op == 'infer'
                            else model._finish_streaming_steps(state, command['max_tokens']))
                        result, metrics = await executor.execute(key, generator)
                    else:
                        # Exact legacy public API, scalar SamplingParams, no batch
                        # executor or collection deadline. The sole worker calls it
                        # synchronously, so different sessions execute serially.
                        count = ((len(state.buffer) + len(audio)) // state.chunk_size_samples
                                 if op == 'infer' else int(len(state.buffer) > 0))
                        began = time.perf_counter()
                        result = (model.streaming_transcribe_no_reset(
                            audio, state, command['max_tokens'], command['rollback']) if op == 'infer'
                            else model.finish_streaming_transcribe_no_reset(state, command['max_tokens']))
                        cost = (time.perf_counter() - began) * 1000 if count else 0.
                        executor.batches += count
                        executor.items += count
                        executor.max_observed_batch = max(executor.max_observed_batch, int(count > 0))
                        metrics = dict(batch_size=int(count > 0), batch_infer_ms=cost,
                                       batch_wait_ms=0., scheduler_queue_ms=0.)
                    value = dict(text=state.text, fixed_text=result[1] if op == 'infer' else result,
                                 vad_end=vad_end, vad_cost_ms=vad_ms, **metrics)
                    value['execution_mode'] = 'batch' if batch_enabled else 'serial'
                    value['infer_ms_scope'] = 'shared_batch' if batch_enabled else 'single_request'
                    value['batch_total_items'] = executor.items
                    value['batch_total_calls'] = executor.batches
                    value['max_observed_batch'] = executor.max_observed_batch
            outgoing.put(dict(id=request_id, value=value))
        except asyncio.CancelledError:
            outgoing.put(dict(id=request_id, error='Session cancelled'))
            raise
        except Exception as exc:
            # A failed prepare/decode may have mutated state. Never retry it.
            sessions.pop(key, None)
            outgoing.put(dict(id=request_id, error=f'{type(exc).__name__}: {exc}'))
            logging.exception('R2T2 batch operation failed session=%s op=%s', key, op)

    try:
        while True:
            commands = [await asyncio.to_thread(incoming.get)]
            # Drain already-arrived IPC commands before executing CPU preparation.
            # Otherwise sequential VAD work can fragment a ready cohort into tiny batches.
            while len(commands) < 128:
                try:
                    commands.append(incoming.get_nowait())
                except queue.Empty:
                    break
            for command in commands:
                if command['op'] == 'stop':
                    return
                key = command['key']
                if key in tasks and not tasks[key].done():
                    if command['op'] == 'close':
                        tasks[key].cancel()
                        await asyncio.gather(tasks[key], return_exceptions=True)
                    else:
                        outgoing.put(dict(id=command['id'], error='Session already has an active operation'))
                        continue
                task = asyncio.create_task(dispatch(command))
                tasks[key] = task
                def forget(done, session_key=key):
                    if tasks.get(session_key) is done:
                        tasks.pop(session_key, None)
                task.add_done_callback(forget)
    finally:
        for task in tasks.values():
            task.cancel()
        await asyncio.gather(*tasks.values(), return_exceptions=True)
        if runner:
            runner.cancel()
            await asyncio.gather(runner, return_exceptions=True)


def model_worker(incoming, outgoing, max_size, wait_ms, batch_enabled=True):
    try:
        asyncio.run(_worker_loop(incoming, outgoing, max_size, wait_ms, batch_enabled))
    except BaseException:
        outgoing.put(dict(id='startup', error=traceback.format_exc()))
        raise


class ModelProcess:
    def __init__(self, max_size=8, wait_ms=5, batch_enabled=True):
        self.max_size, self.wait_ms = max_size, wait_ms
        self.batch_enabled = batch_enabled
        context = mp.get_context('spawn')
        self.incoming, self.outgoing = context.Queue(128), context.Queue(128)
        self.process = context.Process(target=model_worker,
                                       args=(self.incoming, self.outgoing, max_size, wait_ms, batch_enabled))
        self.pending = {}
        self.reader = None

    async def start(self):
        ready = asyncio.get_running_loop().create_future()
        self.pending['startup'] = ready
        self.process.start()
        self.reader = asyncio.create_task(self._read())
        try:
            await asyncio.wait_for(ready, 900)
        except BaseException:
            await self.close()
            raise

    async def _read(self):
        try:
            while True:
                try:
                    message = await asyncio.to_thread(self.outgoing.get, True, 1)
                except queue.Empty:
                    if not self.process.is_alive():
                        raise RuntimeError('R2T2 model worker exited')
                    continue
                future = self.pending.pop(message['id'], None)
                if future and not future.done():
                    if 'error' in message:
                        future.set_exception(RuntimeError(message['error']))
                    else:
                        future.set_result(message['value'])
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            for future in self.pending.values():
                if not future.done():
                    future.set_exception(exc)
            self.pending.clear()

    async def call(self, key, op, **fields):
        if not self.process.is_alive():
            raise RuntimeError('Model worker is unavailable')
        request_id = uuid.uuid4().hex
        future = asyncio.get_running_loop().create_future()
        self.pending[request_id] = future
        try:
            await asyncio.to_thread(self.incoming.put, dict(id=request_id, key=key, op=op, **fields), True, 5)
            return await asyncio.wait_for(future, 120)
        finally:
            self.pending.pop(request_id, None)

    async def close(self):
        if self.process.is_alive():
            with contextlib.suppress(queue.Full):
                self.incoming.put_nowait(dict(op='stop'))
            await asyncio.to_thread(self.process.join, 10)
            if self.process.is_alive():
                self.process.terminate()
                await asyncio.to_thread(self.process.join, 5)
        if self.reader:
            self.reader.cancel()
            await asyncio.gather(self.reader, return_exceptions=True)
        for future in self.pending.values():
            if not future.done():
                future.set_exception(RuntimeError('Model process stopped'))
        self.pending.clear()
        for channel in (self.incoming, self.outgoing):
            channel.cancel_join_thread()
            channel.close()
