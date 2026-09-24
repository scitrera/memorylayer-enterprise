# SPDX-License-Identifier: AGPL-3.0-only
"""Actual pipe ownership/cancellation tests with a tiny stand-in subprocess."""
import asyncio
import sys

import pytest

from memorylayer_embed_server_enterprise.services.ner_worker import LocalGLiNERWorker


def worker(tmp_path):
    script = tmp_path / 'worker.py'
    script.write_text('''import json,sys,time
print('{"ready":true}',flush=True)
for line in sys.stdin:
    request=json.loads(line)
    time.sleep(.03)
    if request['texts']==['invalid']:
        result={'invalid':'too long'}
    else:
        result={'results':[{'person':[text]} for text in request['texts']]}
    print(json.dumps(result),flush=True)
''')
    return LocalGLiNERWorker(model_name='test', default_labels=['person'], python=sys.executable, script=str(script))


async def test_cancelled_request_drains_response_before_next_owner(tmp_path):
    service = worker(tmp_path)
    await service.preload()
    try:
        first = asyncio.create_task(service.extract_batch(['first']))
        while not service._active:
            await asyncio.sleep(0)
        first.cancel()
        second = asyncio.create_task(service.extract_batch(['second']))
        result = await asyncio.gather(first, second, return_exceptions=True)
        assert isinstance(result[0], asyncio.CancelledError)
        assert result[1] == [{'person': ['second']}]
        assert service.get_load_snapshot()['in_flight'] == 0
    finally:
        await service.shutdown()
    assert not service.is_ready and service.process.returncode is not None


async def test_validation_does_not_poison_worker_and_dead_process_is_unready(tmp_path):
    service = worker(tmp_path)
    await service.preload()
    try:
        with pytest.raises(ValueError, match='too long'):
            await service.extract_batch(['invalid'])
        assert await service.extract_batch(['valid']) == [{'person': ['valid']}]
        service.process.kill()
        await service.process.wait()
        assert not service.is_ready
    finally:
        await service.shutdown()
