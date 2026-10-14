"""onpolicy/sim_throttle.py against a mock upstream (CPU only, no API): a 429 pauses all traffic (nothing reaches
the upstream during the pause), a block that persists doubles the pause, the rate drops to 0.7x, every request still
succeeds afterwards, and a request whose client has given up is not sent again.
    ../.venv-client/bin/python tests/test_sim_throttle.py
"""
import asyncio, os, socket, subprocess, sys, time
import aiohttp
from aiohttp import web

HERE = os.path.dirname(os.path.abspath(__file__))
SEEN = []          # (time, status) of every request the upstream received
BLOCK = [0.0, 0.0]  # upstream answers 429 between these monotonic times


async def chat(request):
    now = time.monotonic()
    st = 429 if BLOCK[0] <= now < BLOCK[1] else 200
    SEEN.append((now, st))
    return web.json_response({'choices': [{'message': {'content': 'ok'}, 'finish_reason': 'stop'}]}, status=st)


def free_port():
    s = socket.socket(); s.bind(('127.0.0.1', 0)); p = s.getsockname()[1]; s.close(); return p


async def main():
    app = web.Application(); app.router.add_post('/v1/chat/completions', chat)
    runner = web.AppRunner(app); await runner.setup()
    up = free_port(); await web.TCPSite(runner, '127.0.0.1', up).start()
    port = free_port()
    env = dict(os.environ, OPENAI_API_BASE='http://127.0.0.1:%d/v1' % up, OPENAI_API_KEY='test')
    proc = subprocess.Popen([sys.executable, os.path.join(HERE, '..', 'onpolicy', 'sim_throttle.py'), '--port', str(port),
                             '--rps', '5', '--burst', '2', '--pause', '0.5', '--raise-every', '100'], env=env)
    url = 'http://127.0.0.1:%d' % port
    body = dict(model='m', messages=[{'role': 'user', 'content': 'x'}])
    try:
        async with aiohttp.ClientSession() as s:
            for _ in range(50):
                try:
                    async with s.get(url + '/stats') as r:
                        await r.json(); break
                except aiohttp.ClientError:
                    await asyncio.sleep(0.1)

            async def call(timeout=30):
                async with s.post(url + '/v1/chat/completions', json=body, timeout=aiohttp.ClientTimeout(total=timeout)) as r:
                    return r.status

            # 1. a block that persists for 0.8 s: pause 0.5 s, still blocked -> pause 1.0 s; then all 6 requests succeed
            t0 = time.monotonic(); BLOCK[:] = [t0, t0 + 0.8]; SEEN.clear()
            st = await asyncio.gather(*[call() for _ in range(6)])
            async with s.get(url + '/stats') as r:
                js = await r.json()
            assert all(x == 200 for x in st) and js['blocks'] == 2, (st, js)
            n429 = [t for t, c in SEEN if c == 429]
            # the burst (2) sends two at once; their two 429s are one block; then one probe per pause, no retry stream
            assert len(n429) == 3 and n429[1] - n429[0] < 0.1, SEEN
            gaps = [round(b - a, 2) for a, b in zip([t for t, _ in SEEN][1:], [t for t, _ in SEEN][2:])]
            assert gaps[0] >= 0.45 and gaps[1] >= 0.95, gaps  # nothing reached the upstream during the pauses
            assert abs(js['rate'] - 5 * 0.7 * 0.7) < 1e-6, js
            print('pause / doubling / AIMD ok: upstream saw %d requests (%d x 429), gaps %s, rate %.2f' % (
                len(SEEN), len(n429), [round(g, 2) for g in gaps[:3]], js['rate']))

            # 2. a client that gives up while its request waits in a pause: the request is not sent again
            await asyncio.sleep(1.2)
            t0 = time.monotonic(); BLOCK[:] = [t0, t0 + 0.1]; SEEN.clear()
            try:
                await call(timeout=0.3)
                raise AssertionError('expected a client timeout')
            except asyncio.TimeoutError:
                pass
            await asyncio.sleep(2.5)
            assert len(SEEN) == 1 and SEEN[0][1] == 429, SEEN
            print('abandoned request not resent ok')
    finally:
        proc.terminate(); proc.wait()
        await runner.cleanup()
    print('sim_throttle test passed')


asyncio.run(main())
