# SPDX-License-Identifier: AGPL-3.0-only
"""Owned GLiNER subprocess facade for an independently locked Python environment."""
import asyncio
import json
import os


class LocalGLiNERWorker:
    def __init__(self, *, model_name, default_labels, python, script):
        self.model_name, self.default_labels = model_name, default_labels
        self.python, self.script = python, script
        self.process = None
        self._ready = False
        self._lock = asyncio.Lock()
        self._pending = 0
        self._active = 0

    @property
    def is_ready(self):
        return self._ready and self.process is not None and self.process.returncode is None

    async def preload(self):
        async with self._lock:
            if self.is_ready:
                return
            self.process = await asyncio.create_subprocess_exec(self.python, self.script,
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                limit=1024 * 1024, start_new_session=True,
                env={**os.environ, 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'UV_OFFLINE': '1'})
            try:
                async with asyncio.timeout(180):
                    reply = json.loads(await self.process.stdout.readline())
                if reply != {'ready': True}:
                    raise RuntimeError('NER worker failed readiness handshake')
                self._ready = True
            except BaseException:
                await self.shutdown()
                raise

    async def extract_batch(self, texts, labels=None):
        if not self.is_ready:
            raise RuntimeError('NER worker unavailable')
        if not texts:
            return []
        if self._pending >= int(os.environ.get("MEMORYLAYER_EMBED_MAX_CONCURRENT", "8")):
            raise RuntimeError('NER worker queue full')
        self._pending += 1
        try:
            async with self._lock:
                self._active = 1
                task = asyncio.create_task(self._exchange(texts, labels or self.default_labels))
                try:
                    return await asyncio.shield(task)
                except asyncio.CancelledError:
                    # Drain the reply before another request can own the pipe.
                    while not task.done():
                        try:
                            await asyncio.shield(task)
                        except asyncio.CancelledError:
                            continue
                    task.result()
                    raise
                finally:
                    self._active = 0
        finally:
            self._pending -= 1

    async def _exchange(self, texts, labels):
        try:
            async with asyncio.timeout(120):
                self.process.stdin.write((json.dumps({'texts': texts, 'labels': labels}) + '\n').encode())
                await self.process.stdin.drain()
                reply = json.loads(await self.process.stdout.readline())
        except (TimeoutError, OSError, ValueError):
            await self.shutdown()
            raise RuntimeError('NER worker transport failed') from None
        if 'invalid' in reply:
            raise ValueError(reply['invalid'])
        if 'error' in reply:
            raise RuntimeError(reply['error'])
        if len(reply['results']) != len(texts):
            raise RuntimeError('NER worker returned incorrect batch length')
        return reply['results']

    async def shutdown(self):
        self._ready = False
        if self.process is not None and self.process.returncode is None:
            self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), 15)
            except TimeoutError:
                self.process.kill()
                await self.process.wait()

    def get_model_info(self):
        return {'model_name': self.model_name, 'loaded': self.is_ready, 'runtime': 'local_python_worker', 'dtype': 'float32'}

    def get_load_snapshot(self):
        return {'in_flight': self._active, 'queued': self._pending - self._active,
                'max_concurrent': 1, 'utilization': float(self._active)}
