"""Adaptive rate limiter in front of an OpenAI-compatible API for the multi-turn user / tool simulator: one per API
key, each on its own machine; 127.0.0.1:<port>/v1/* -> $OPENAI_API_BASE/*, the key added.

Token bucket with retries on 429 / 5xx / connection errors. The upstream API may rate-limit per key and per source IP,
its limit is shared with any other use of the key, and a forwarder that keeps retrying at a fixed rate while blocked
only prolongs the block. Here:
  - a 429 pauses ALL of this key's traffic: --pause s (60), doubled for each further block within 10 minutes, up to
    300 s; 429s of requests already in flight when the pause began do not count again;
  - AIMD rate: each block cuts the rate to 0.7x (not below --min-rps); every 5 minutes without a block it climbs back
    by 0.05/s, up to --rps;
  - a request whose client has gone (the simulator call timed out and was retried) is not sent again and uses no
    token (checked while it waits, before it takes one).
GET /stats -> ok / retried / failed / blocks / rate / paused_s.
    python sim_throttle.py --port 8290 --rps 1.0      (OPENAI_API_BASE / OPENAI_API_KEY from the environment)
"""
import argparse
import asyncio
import os
import time

import aiohttp
from aiohttp import web

ap = argparse.ArgumentParser()
ap.add_argument('--port', type=int, default=8290)
ap.add_argument('--rps', type=float, default=1.0, help='highest sustained requests per second to the upstream')
ap.add_argument('--min-rps', type=float, default=0.3)
ap.add_argument('--burst', type=int, default=2)
ap.add_argument('--pause', type=float, default=60.0, help='seconds all traffic waits after a 429 (doubles, max 300)')
ap.add_argument('--raise-every', type=float, default=300.0, help='seconds without a block before the rate rises')
ap.add_argument('--give-up', type=float, default=3600.0, help='seconds of retrying before returning the error')
a = ap.parse_args()
UP = os.environ['OPENAI_API_BASE'].rstrip('/')
KEY = os.environ['OPENAI_API_KEY']
stats = {'ok': 0, 'retried': 0, 'failed': 0, 'blocks': 0}


class Limiter:
    def __init__(self):
        self.rate, self.tokens, self.t = a.rps, float(a.burst), time.monotonic()
        self.lock = asyncio.Lock()
        self.paused_until, self.pause, self.last_block, self.last_raise = 0.0, 0.0, -1e9, time.monotonic()

    async def take(self, is_gone=None):
        """wait for a token; False (no token used) if is_gone() says the client has given up meanwhile"""
        async with self.lock:
            while True:
                if is_gone is not None and is_gone():
                    return False
                now = time.monotonic()
                if now < self.paused_until:
                    await asyncio.sleep(self.paused_until - now)
                    self.t = time.monotonic()  # no tokens accrue during a pause: one request goes first
                    continue
                if self.rate < a.rps and now - max(self.last_block, self.last_raise) >= a.raise_every:
                    self.rate, self.last_raise = min(a.rps, self.rate + 0.05), now
                self.tokens = min(a.burst, self.tokens + (now - self.t) * self.rate)
                self.t = now
                if self.tokens >= 1:
                    self.tokens -= 1
                    return True
                await asyncio.sleep((1 - self.tokens) / self.rate)

    def blocked(self, sent_at):
        if sent_at < self.last_block:  # sent before the current block was noticed: same block
            return
        now = time.monotonic()
        self.pause = min(300.0, self.pause * 2) if now - self.last_block < 600 else a.pause
        self.paused_until, self.last_block = now + self.pause, now
        self.rate, self.tokens = max(a.min_rps, self.rate * 0.7), 0.0
        stats['blocks'] += 1
        print('[%s] 429: all traffic paused %.0f s, rate now %.2f/s (%s)' % (time.strftime('%H:%M:%S'), self.pause,
                                                                           self.rate, stats), flush=True)


lim = Limiter()


def gone(request):
    return request.transport is None or request.transport.is_closing()


async def forward(request):
    body = await request.read()
    path = request.match_info['tail']
    t0, attempt = time.monotonic(), 0
    while True:
        # the client gave up on this request (its timeout) while it waited: do not spend a token or the quota on it
        if not await lim.take(lambda: gone(request)):
            return web.Response(status=499)
        sent = time.monotonic()
        try:
            async with request.app['session'].post(f'{UP}/{path}', data=body, headers={
                    'Authorization': f'Bearer {KEY}', 'Content-Type': 'application/json'}) as r:
                data = await r.read()
                if r.status == 429:
                    lim.blocked(sent)
                    err = 'HTTP 429'
                elif r.status in (500, 502, 503, 504, 529):
                    err = f'HTTP {r.status}'
                else:
                    stats['ok'] += 1
                    return web.Response(body=data, status=r.status, content_type='application/json')
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            err = type(e).__name__
        attempt += 1
        stats['retried'] += 1
        if time.monotonic() - t0 > a.give_up:
            stats['failed'] += 1
            return web.Response(body=b'{"error": {"message": "sim_throttle: gave up after retries: %s"}}' % err.encode(),
                                status=429, content_type='application/json')
        if err != 'HTTP 429':  # 5xx / connection error: this request backs off; a 429 waits in the limiter's pause
            await asyncio.sleep(min(60, 5 * attempt))


async def health(request):
    return web.json_response(dict(stats, rate=round(lim.rate, 3),
                                  paused_s=round(max(0.0, lim.paused_until - time.monotonic()), 1)))


async def make_app():
    app = web.Application(client_max_size=64 * 1024 * 1024)
    app['session'] = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=900))
    app.router.add_post('/v1/{tail:.*}', forward)
    app.router.add_get('/stats', health)
    return app


if __name__ == '__main__':
    web.run_app(make_app(), host='127.0.0.1', port=a.port, print=None)
