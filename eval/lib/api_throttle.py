"""Local rate-limiting forwarder for the tau2 user simulator and judge: 127.0.0.1:<port>/v1/* -> $QADX_THROTTLE_UPSTREAM/*.

Endpoints behind a per-key rate rule block a key for minutes after a burst (two or three tau2 processes x 48
simulations all open with a user-simulator call), and tau2 then fails those simulations as infrastructure errors
(--max-retries 0 on purpose: re-running only the failed simulations would bias the results). This forwarder keeps the
request rate under --rps with a token bucket shared by all tau2 processes of a phase, and on 429 / 5xx / connection
errors waits (30 s, 60 s, ... up to 10 min) and retries, for up to --give-up seconds, so an API hiccup delays a
simulation instead of failing it. Only the Authorization header and the body are forwarded.
    QADX_THROTTLE_UPSTREAM=https://host/v1 QADX_THROTTLE_KEY=... python api_throttle.py --port 8190 --rps 1.2
"""

import argparse
import asyncio
import os
import time

import aiohttp
from aiohttp import web

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8190)
ap.add_argument("--rps", type=float, default=1.2, help="sustained requests per second to the upstream")
ap.add_argument("--burst", type=int, default=4)
ap.add_argument("--give-up", type=float, default=3 * 3600, help="seconds of retrying before returning the error")
a = ap.parse_args()
UP = os.environ["QADX_THROTTLE_UPSTREAM"].rstrip("/")
KEY = os.environ["QADX_THROTTLE_KEY"]


class Bucket:
    def __init__(self, rate, burst):
        self.rate, self.cap, self.tokens, self.t = rate, burst, float(burst), time.monotonic()
        self.lock = asyncio.Lock()

    async def take(self):
        async with self.lock:
            while True:
                now = time.monotonic()
                self.tokens = min(self.cap, self.tokens + (now - self.t) * self.rate)
                self.t = now
                if self.tokens >= 1:
                    self.tokens -= 1
                    return
                await asyncio.sleep((1 - self.tokens) / self.rate)


bucket = Bucket(a.rps, a.burst)
stats = {"ok": 0, "retried": 0, "failed": 0}


async def forward(request):
    body = await request.read()
    path = request.match_info["tail"]
    t0, attempt = time.monotonic(), 0
    while True:
        await bucket.take()
        try:
            async with request.app["session"].post(f"{UP}/{path}", data=body, headers={
                    "Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}) as r:
                data = await r.read()
                if r.status not in (429, 500, 502, 503, 504, 529):
                    stats["ok"] += 1
                    return web.Response(body=data, status=r.status, content_type="application/json")
                err = f"HTTP {r.status}"
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            err = type(e).__name__
        attempt += 1
        stats["retried"] += 1
        if time.monotonic() - t0 > a.give_up:
            stats["failed"] += 1
            return web.Response(body=b'{"error": {"message": "api_throttle: gave up after retries: %s"}}' % err.encode(),
                                status=429, content_type="application/json")
        wait = min(600, 30 * attempt)
        print(f"[{time.strftime('%H:%M:%S')}] {err} on {path}; retry {attempt} in {wait}s ({stats})", flush=True)
        await asyncio.sleep(wait)


async def health(request):
    return web.json_response(stats)


async def make_app():
    app = web.Application(client_max_size=64 * 1024 * 1024)
    app["session"] = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=900))
    app.router.add_post("/v1/{tail:.*}", forward)
    app.router.add_get("/stats", health)
    return app


if __name__ == "__main__":
    web.run_app(make_app(), host="127.0.0.1", port=a.port, print=None)
