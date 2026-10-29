"""CPU-only batching, algorithm, fairness and IPC tests; no model import."""
import ast
import asyncio
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import re
import sys
from types import SimpleNamespace, ModuleType
from typing import Tuple
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from r2t2_batch import BatchExecutor, ModelProcess, _worker_loop, parse_batch_enabled, configure_geometry


class ConfigTests(unittest.TestCase):
    def test_explicit_switch_defaults_and_invalid_values(self):
        for value in ('1', 'true', ' TRUE '):
            self.assertTrue(parse_batch_enabled(value))
        for value in ('0', 'false', ' FALSE '):
            self.assertFalse(parse_batch_enabled(value))
        with self.assertRaises(ValueError):
            parse_batch_enabled('auto')

    def test_geometry_480_and_invalid_values(self):
        import os
        with patch.dict(os.environ, R2T2_CHUNK_MS='480', R2T2_LOOKAHEAD_MS='160'):
            config = SimpleNamespace()
            self.assertEqual(configure_geometry(config), (480, 160))
            self.assertEqual(config.CHUNK_ASR_SIZE, 7680)
            self.assertEqual(config.CHUNK_ASR_SECONDS, .48)
        for value in ('0', '481', '-160', 'abc'):
            with patch.dict(os.environ, R2T2_CHUNK_MS=value):
                with self.assertRaises(ValueError):
                    configure_geometry(SimpleNamespace())


def fake_worker(incoming, outgoing, *_):
    outgoing.put(dict(id='startup', value=dict(ready=True)))
    while True:
        command = incoming.get()
        if command['op'] == 'stop':
            return
        if command['op'] == 'die':
            return
        outgoing.put(dict(id=command['id'], value=dict(key=command['key'], op=command['op'])))


def harness():
    source = ast.parse((ROOT / 'r2t2/r2t2_asr.py').read_text(encoding='utf-8'))
    model = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == 'R2T2ASRModel')
    names = {'_append_stream_audio', '_run_stream_steps', 'streaming_transcribe_no_reset', '_streaming_transcribe_steps',
             'finish_streaming_transcribe_no_reset', '_finish_streaming_steps'}
    nodes = [n for n in model.body if isinstance(n, ast.FunctionDef) and n.name in names]
    klass = ast.ClassDef(name='Harness', bases=[], keywords=[], body=nodes, decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[klass], type_ignores=[]))
    namespace = dict(np=np, Tuple=Tuple, ASRStreamingState=SimpleNamespace, SAMPLE_RATE=16000, re=re,
                     _normalize_punct_by_context=lambda t: t,
                     parse_language_output=lambda t, **kw: ('Chinese', t),
                     parse_asr_output=lambda t, **kw: ('Chinese', t))
    exec(compile(module, '<actual-streaming-methods>', 'exec'), namespace)
    model = namespace['Harness']()
    model.backend = 'vllm'
    model.sampling_params = SimpleNamespace(max_tokens=1)
    model.processor = SimpleNamespace(tokenizer=SimpleNamespace(encode=lambda t: list(map(ord, t)),
                                                               decode=lambda ids: ''.join(map(chr, ids))))
    return model


def state():
    return SimpleNamespace(buffer=np.empty(0, np.float32), audio_accum=np.empty(0, np.float32),
        chunk_size_samples=2560, chunk_size_sec=.16, _first_chunk_discarded=True,
        chunk_text=[], force_language='Chinese', language='', prompt_raw='', _raw_decoded='',
        unfixed_token_num=0, text='', chunk_id=0, last_fixed_text='', chunk_sample_lengths=[])


def outputs(inputs, params):
    return [SimpleNamespace(outputs=[SimpleNamespace(text='甲' if item['multi_modal_data']['audio'][0][0] > 0 else '乙')])
            for item in inputs]


class SchedulerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.calls = []
        def generate(inputs, params):
            self.calls.append((inputs, params))
            return [f'{i}:{p}' for i, p in zip(inputs, params)]
        self.scheduler = BatchExecutor(generate, 8, 5)
        self.runner = asyncio.create_task(self.scheduler.run())

    async def asyncTearDown(self):
        self.runner.cancel()
        await asyncio.gather(self.runner, return_exceptions=True)

    async def test_eight_sessions_batch_and_preserve_parameters(self):
        def steps(i):
            value = yield (i, i + 1)
            return value[0]
        values = await asyncio.gather(*(self.scheduler.execute(str(i), steps(i)) for i in range(8)))
        self.assertEqual([v[0] for v in values], [f'{i}:{i+1}' for i in range(8)])
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(len(self.calls[0][0]), 8)

    async def test_same_session_sequential_steps_rejoin_fairly(self):
        def steps(key):
            first = yield (key + '1', 1)
            second = yield (key + '2', 2)
            return first + second
        await asyncio.gather(*(self.scheduler.execute(k, steps(k)) for k in ('A', 'B')))
        self.assertEqual([c[0] for c in self.calls], [['A1', 'B1'], ['A2', 'B2']])

    async def test_one_session_does_not_wait_for_eight(self):
        def steps():
            yield ('one', 1)
            return 'done'
        value, metrics = await asyncio.wait_for(self.scheduler.execute('one', steps()), .5)
        self.assertEqual(value, 'done')
        self.assertEqual(metrics['batch_size'], 1)

    async def test_duplicate_session_rejected(self):
        def steps():
            yield ('one', 1)
        first = asyncio.create_task(self.scheduler.execute('same', steps()))
        await asyncio.sleep(0)
        with self.assertRaises(RuntimeError):
            await self.scheduler.execute('same', steps())
        await first

    async def test_batch_failure_is_not_replayed(self):
        self.scheduler.generate = lambda *args: (_ for _ in ()).throw(ValueError('engine failed'))
        def steps():
            yield ('one', 1)
        with self.assertRaisesRegex(ValueError, 'engine failed'):
            await self.scheduler.execute('one', steps())
        self.assertFalse(self.scheduler.busy)

    async def test_cancelled_job_does_not_apply_output(self):
        applied = []
        def steps():
            yield ('one', 1)
            applied.append(True)
        task = asyncio.create_task(self.scheduler.execute('one', steps()))
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await asyncio.sleep(.02)
        self.assertFalse(applied)


class AlgorithmTests(unittest.IsolatedAsyncioTestCase):
    async def test_480_window_audio_text_alignment_across_multiple_rollovers(self):
        model = harness()
        current = state()
        received = []
        next_sample = 0
        drops = []
        for size in [10240] + [7680] * 80:
            chunk = np.arange(next_sample, next_sample + size, dtype=np.float32)
            before = len(current.audio_accum)
            model._append_stream_audio(current, chunk)
            current.chunk_text.append(str(len(received)))
            current.chunk_sample_lengths.append(size)
            received.append(size)
            next_sample += size
            removed = before + size - len(current.audio_accum)
            if removed:
                drops.append(removed)
            self.assertEqual(sum(current.chunk_sample_lengths), len(current.audio_accum))
            self.assertEqual(len(current.chunk_text), len(current.chunk_sample_lengths))
            np.testing.assert_array_equal(current.audio_accum,
                np.arange(next_sample - len(current.audio_accum), next_sample, dtype=np.float32))
        self.assertGreaterEqual(len(drops), 3)
        self.assertEqual(drops[0], 133120)  # 640ms + 16*480ms = 8.32s.
        self.assertTrue(all(n == 130560 for n in drops[1:]))  # 17*480ms = 8.16s.
        self.assertLessEqual(len(current.audio_accum), 16 * 16000)

    async def test_160_first_320_window_retains_original_eight_second_boundary(self):
        model, current = harness(), state()
        for size in [5120] + [2560] * 100:
            model._append_stream_audio(current, np.ones(size, dtype=np.float32))
            current.chunk_text.append('甲')
            current.chunk_sample_lengths.append(size)
        self.assertEqual(len(current.audio_accum), (5120 + 100*2560) - 128000)
        self.assertEqual(len(current.chunk_text), 101 - 49)
    async def test_real_generators_isolate_state_and_flush_tail(self):
        vllm = ModuleType('vllm')
        vllm.SamplingParams = lambda **kw: SimpleNamespace(**kw)
        with patch.dict(sys.modules, vllm=vllm):
            model = harness()
            calls = []
            def generate(inputs, params):
                calls.append(len(inputs))
                return outputs(inputs, params)
            scheduler = BatchExecutor(generate)
            runner = asyncio.create_task(scheduler.run())
            a, b = state(), state()
            try:
                # A single submission containing 3 chunks must be 3 distinct batches.
                values = await asyncio.gather(
                    scheduler.execute('A', model._streaming_transcribe_steps(np.ones(7680, np.float32), a, 2)),
                    scheduler.execute('B', model._streaming_transcribe_steps(-np.ones(7680, np.float32), b, 3)))
                self.assertEqual(a.last_fixed_text, '甲甲甲')
                self.assertEqual(b.last_fixed_text, '乙乙乙')
                self.assertEqual(calls, [2, 2, 2])
                before = len(calls)
                await scheduler.execute('A', model._streaming_transcribe_steps(np.ones(100, np.float32), a, 2))
                self.assertEqual(len(calls), before)
                value, _ = await scheduler.execute('A', model._finish_streaming_steps(a, 2))
                self.assertEqual(value, '甲甲甲甲')
                self.assertEqual(len(a.buffer), 0)
                before = len(calls)
                await scheduler.execute('A', model._finish_streaming_steps(a, 2))
                self.assertEqual(len(calls), before)  # Empty EOS must not decode again.
            finally:
                runner.cancel()
                await asyncio.gather(runner, return_exceptions=True)

    async def test_synchronous_api_and_batch_are_equivalent_across_window_rollover(self):
        vllm = ModuleType('vllm')
        vllm.SamplingParams = lambda **kw: SimpleNamespace(**kw)
        with patch.dict(sys.modules, vllm=vllm):
            model = harness()
            model.model = SimpleNamespace(generate=lambda items, sampling_params, **kw: outputs(items, sampling_params))
            a, b = state(), state()
            audio = np.ones(17 * 16000 + 100, np.float32)
            serial = model.streaming_transcribe_no_reset(audio, a, 2)
            scheduler = BatchExecutor(outputs, wait_ms=0)
            runner = asyncio.create_task(scheduler.run())
            try:
                batch, _ = await scheduler.execute('B', model._streaming_transcribe_steps(audio, b, 2))
                self.assertEqual(batch, serial)
                self.assertEqual(a.chunk_text, b.chunk_text)
                np.testing.assert_array_equal(a.audio_accum, b.audio_accum)
                np.testing.assert_array_equal(a.buffer, b.buffer)
                self.assertLess(len(b.audio_accum), 16 * 16000)
            finally:
                runner.cancel()
                await asyncio.gather(runner, return_exceptions=True)


class IPCTests(unittest.IsolatedAsyncioTestCase):
    async def test_spawn_process_replies_and_failure_propagates(self):
        client = ModelProcess()
        client.process = mp.get_context('spawn').Process(target=fake_worker,
            args=(client.incoming, client.outgoing, 8, 5))
        try:
            await client.start()
            a, b = await asyncio.gather(client.call('a', 'open'), client.call('b', 'open'))
            self.assertEqual(a['key'], 'a')
            self.assertEqual(b['key'], 'b')
            with self.assertRaisesRegex(RuntimeError, 'worker exited'):
                await client.call('a', 'die')
        finally:
            await client.close()


class ProtocolTests(unittest.IsolatedAsyncioTestCase):
    async def test_original_packet_geometry_and_short_eos_tail(self):
        await self.run_geometry(160, 7780)

    async def test_480_packet_geometry_and_short_eos_tail(self):
        await self.run_geometry(480, 320100)

    async def run_geometry(self, chunk_ms, samples):
        source = ast.parse((ROOT / 'ws_batch_server.py').read_text(encoding='utf-8'))
        node = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == 'ProtocolSession')
        legacy = SimpleNamespace(MAX_TOKENS=10, STEP_MS=chunk_ms, LOOKAHEAD_MS=160, CHUNK_ASR_SECONDS=chunk_ms/1000,
            CHUNK_ASR_SIZE=chunk_ms*16, UNFIX_TOKEN_NUM=0, resolve_qwen_context=lambda *args: '',
            split_text_to_tokens=list, is_last_token_chinese=lambda t: True,
            detect_hallucination=lambda *args: (False, ''))
        namespace = dict(np=np, uuid=__import__('uuid'), deque=__import__('collections').deque,
                         time=__import__('time'), RATE=16000, legacy=legacy, json=json)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), '<protocol>', 'exec'), namespace)
        model = harness()
        sessions = {}
        scheduler = BatchExecutor(outputs)
        runner = asyncio.create_task(scheduler.run())
        class Client:
            async def call(self, key, op, **fields):
                if op == 'open':
                    sessions[key] = state()
                    return {}
                current = sessions[key]
                if op == 'infer':
                    current.chunk_size_samples = fields['chunk_samples']
                    gen = model._streaming_transcribe_steps(np.frombuffer(fields['audio'], dtype='<f4'), current, fields['max_tokens'])
                else:
                    gen = model._finish_streaming_steps(current, fields['max_tokens'])
                result, metrics = await scheduler.execute(key, gen)
                return dict(text=current.text, fixed_text=result[1] if op == 'infer' else result,
                    vad_end=False, vad_cost_ms=0, batch_total_items=0, batch_total_calls=0,
                    max_observed_batch=1, **metrics)
        emitted = []
        async def send(raw):
            emitted.append(json.loads(raw))
        vllm = ModuleType('vllm')
        vllm.SamplingParams = lambda **kw: SimpleNamespace(**kw)
        try:
            with patch.dict(sys.modules, vllm=vllm):
                s = namespace['ProtocolSession'](Client(), dict(requestId='call'), send)
                await s.open()
                await s.feed(np.ones(7680, np.float32), 0)
                self.assertEqual(s.processed, 7680 if chunk_ms == 160 else 0)
                self.assertEqual(len([e for e in emitted if e.get('status') == 'success']), 2 if chunk_ms == 160 else 0)
                for start in range(7680, samples, 7680):
                    await s.feed(np.ones(min(7680, samples-start), np.float32), 0)
                await s.finish()
                self.assertEqual(s.processed, samples)
                self.assertEqual(emitted[-1]['msg']['text'], '甲')
                self.assertTrue(emitted[-1]['msg']['reset'])
                self.assertEqual(emitted[-1]['msg']['audio_processed_samples'], samples)
                expected_steps = 1 + (samples - (chunk_ms+160)*16) // (chunk_ms*16) + 1
                self.assertEqual(sessions[s.key].chunk_id, expected_steps)
        finally:
            runner.cancel()
            await asyncio.gather(runner, return_exceptions=True)


class WorkerTests(unittest.IsolatedAsyncioTestCase):
    async def test_ipc_cohort_batches_eight_without_disabled_vad_work(self):
        sizes, replies, lists = await self.run_cohort(True, 8)
        self.assertEqual(sizes, [8])
        self.assertTrue(all(lists))
        self.assertTrue(all(r['value']['execution_mode'] == 'batch' for r in replies))

    async def test_batch_mode_is_explicit_even_with_two_sessions(self):
        sizes, replies, lists = await self.run_cohort(True, 2)
        self.assertEqual(sizes, [2])
        self.assertTrue(all(lists))

    async def test_serial_mode_uses_original_scalar_api_with_two_sessions(self):
        sizes, replies, lists = await self.run_cohort(False, 2)
        self.assertEqual(sizes, [1, 1])
        self.assertFalse(any(lists))
        for r in replies:
            self.assertEqual(r['value']['batch_wait_ms'], 0)
            self.assertEqual(r['value']['infer_ms_scope'], 'single_request')

    async def test_serial_mode_stays_serial_with_eight_sessions(self):
        sizes, replies, lists = await self.run_cohort(False, 8)
        self.assertEqual(sizes, [1] * 8)
        self.assertFalse(any(lists))
        self.assertTrue(all(r['value']['execution_mode'] == 'serial' for r in replies))

    async def run_cohort(self, batch_enabled, count):
        import queue
        incoming, outgoing = queue.Queue(), queue.Queue()
        model = harness()
        sizes = []
        parameter_lists = []
        def generate(inputs, sampling_params, **kw):
            sizes.append(len(inputs))
            parameter_lists.append(isinstance(sampling_params, list))
            return outputs(inputs, sampling_params)
        model.model = SimpleNamespace(generate=generate)
        model.init_streaming_state = lambda **kw: state()
        legacy = ModuleType('ws_server')
        legacy.asr_model = model
        # Must never clone or call this template when VAD is disabled.
        legacy.stream_vad = None
        async def initialize(app):
            pass
        legacy.initialize_models = initialize
        vllm = ModuleType('vllm')
        vllm.SamplingParams = lambda **kw: SimpleNamespace(**kw)
        with patch.dict(sys.modules, ws_server=legacy, vllm=vllm):
            worker = asyncio.create_task(_worker_loop(incoming, outgoing, 8, 5, batch_enabled))
            async def response():
                return await asyncio.to_thread(outgoing.get, True, 3)
            try:
                self.assertEqual((await response())['id'], 'startup')
                for i in range(count):
                    incoming.put(dict(id=f'open{i}', key=str(i), op='open', options={}, vad_enabled=False))
                for _ in range(count):
                    self.assertNotIn('error', await response())
                for i in range(count):
                    incoming.put(dict(id=f'infer{i}', key=str(i), op='infer',
                        audio=np.ones(2560, dtype='<f4').tobytes(), chunk_samples=2560,
                        max_tokens=i+1, rollback=False))
                replies = []
                for _ in range(count):
                    reply = await response()
                    replies.append(reply)
                    self.assertNotIn('error', reply)
                    self.assertEqual(reply['value']['batch_size'], count if batch_enabled else 1)
                    self.assertFalse(reply['value']['vad_end'])
                # A short tail must flush through the chosen public/generator path.
                incoming.put(dict(id='tail', key='0', op='infer', audio=np.ones(100, dtype='<f4').tobytes(),
                                  chunk_samples=2560, max_tokens=2, rollback=False))
                self.assertNotIn('error', await response())
                incoming.put(dict(id='finish', key='0', op='finish', max_tokens=2))
                final = await response()
                self.assertNotIn('error', final)
                self.assertEqual(final['value']['fixed_text'], '甲甲')
                # Exclude that final one-item tail decode from cohort assertions.
                return sizes[:-1], replies, parameter_lists[:-1]
            finally:
                incoming.put(dict(op='stop'))
                await asyncio.wait_for(worker, 5)


if __name__ == '__main__':
    unittest.main()
