"""
ShadowV1 by Vortex: your own website + API + Telegram bot on top of the DeepSeek web chat.

  /                      chat website
  /<anything>?key=KEY    prompt URL:  https://HOST/write a poem?key=KEY   (also ?think=1 &search=1)
  /v1/chat/completions   OpenAI-compatible (also without /v1). Works with Cline / Continue / Roo in VS Code
  /v1/images/generations image generation   |   /img/<prompt>?key=KEY
  /pins/<query>?n=5&key= picture search     |   /build, /github/*, /render/*  (needs tokens)
  /guide /connect /remote  help and sign-in pages

Env vars: MY_API_KEY (required), REDIS_URL (keeps your login across restarts), TELEGRAM_BOT_TOKEN,
TELEGRAM_ALLOWED_IDS, GITHUB_TOKEN, RENDER_API_KEY, POLLINATIONS_KEY (optional), IMAGE_MODEL (optional)
"""
import asyncio
import base64
import json
import os
import random
import re
import secrets
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Header, HTTPException, Response
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel
from playwright.async_api import async_playwright
import httpx

from shadow import (BOT_NAME, BOT_MAKER, JS_GRAB, JS_TOGGLE, REMINDER, SEARCH_LABELS, THINK_LABELS,
                    build_convo, clean_html, clean_reply, detect_intent, extract_files, flags_from_model,
                    has_cjk_prose, html_to_md, is_agent_request, md_to_tg, slugify, split_md, with_system)
from media import MediaError, fetch_bytes, generate_image, img_path, search_pictures, sign
import agent as AG
from integrations import GitHub, IntegrationError, RenderAPI

# ---------- CONFIG ----------
API_KEYS = [k.strip() for k in os.getenv("MY_API_KEY", "").split(",") if k.strip()]  # comma = several keys
DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
STATE_PATH = DATA_DIR / "state.json"
REDIS_URL = os.getenv("REDIS_URL", "").strip()
EXTERNAL_URL = os.getenv("RENDER_EXTERNAL_URL", "").rstrip("/")
SITE = "https://chat.deepseek.com"
HERE = Path(__file__).parent

# DeepSeek changes its page often. If replies stop working, inspect the
# page (F12) and update these selectors.
SEL_INPUT = "textarea"
SEL_REPLY = "div.ds-markdown"

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36")
VIEW = {"width": 1000, "height": 700}        # chat page (desktop layout)
LOGIN_VIEW = {"width": 460, "height": 800}   # sign-in page (fits a phone screen)
LAUNCH_ARGS = [
    "--disable-blink-features=AutomationControlled",
    "--no-sandbox",
    "--disable-dev-shm-usage",
    "--disable-gpu",
    "--disable-extensions",
    "--disable-background-networking",
    "--mute-audio",
    "--renderer-process-limit=1",
    "--js-flags=--max-old-space-size=256",
]
ALLOWED_KEYS = {"Enter", "Tab", "Backspace", "Escape", "ArrowDown", "ArrowUp",
                "ArrowLeft", "ArrowRight", "Delete", "Space"}

# Runs inside the DeepSeek page: is there a login token, and whose account?
JS_STATUS = """
async () => {
  const raw = localStorage.getItem('userToken');
  if (!raw) return {connected: false};
  let tok = raw;
  try { const j = JSON.parse(raw); if (j && j.value) tok = j.value; } catch (e) {}
  let account = null, bad = false;
  try {
    const r = await fetch('/api/v0/users/current', {headers: {authorization: 'Bearer ' + tok}});
    if (r.status === 401 || r.status === 403) bad = true;
    else {
      const j = await r.json();
      const d = (j && j.data && (j.data.biz_data || j.data)) || {};
      account = d.email || d.mobile_number || d.mobile ||
                (d.id_profile && d.id_profile.name) || d.name || null;
    }
  } catch (e) {}
  return {connected: !bad, account};
}
"""
FIX_CHINESE = ("Your last reply contained Chinese characters, which are not allowed. Rewrite your full previous "
               "answer in English only, keeping all code unchanged. Output only the rewritten answer.")
BUILD_SUFFIX = ("\n\nBuild this completely now. Do not ask any questions; decide sensible defaults. Output EVERY file "
                "with its full path on the line above its code block. Include a Dockerfile that serves the app on "
                "the port from the PORT environment variable (default 10000), a README.md and a .gitignore.")
# ----------------------------


def log(*a):
    print(*a, flush=True)


class State:
    pw = None
    browser = None
    ctx = None
    page = None
    http = None
    login_pages: list = []
    login_active = False
    lock = asyncio.Lock()
    img_sem = asyncio.Semaphore(2)
    codes: dict = {}                      # one-time /connect codes -> expiry
    status = {"connected": None, "account": None}
    status_at = 0.0
    warm = False                          # page is sitting on an empty new chat
    last_stateless = False
    tasks: list = []


# ---------------- storage: Render Key Value (keeps you logged in) + local file ----------------
class KV:
    r = None

    @classmethod
    async def start(cls):
        if not REDIS_URL:
            log("[store] REDIS_URL not set: login is kept only in local files (lost on redeploy)")
            return
        try:
            import redis.asyncio as aioredis
            cls.r = aioredis.from_url(REDIS_URL, decode_responses=True, socket_timeout=8,
                                      socket_connect_timeout=8)
            await cls.r.ping()
            log("[store] connected to Key Value store")
        except Exception as e:
            cls.r = None
            log("[store] could not connect to REDIS_URL:", type(e).__name__)

    @classmethod
    async def get(cls, key: str):
        if cls.r:
            try:
                v = await cls.r.get(key)
                if v:
                    return v
            except Exception:
                pass
        f = DATA_DIR / (key.replace(":", "_") + ".json")
        return f.read_text() if f.exists() else None

    @classmethod
    async def set(cls, key: str, value: str):
        try:
            (DATA_DIR / (key.replace(":", "_") + ".json")).write_text(value)
        except Exception:
            pass
        if cls.r:
            try:
                await cls.r.set(key, value)
            except Exception:
                pass


async def save_state():
    try:
        state = await State.ctx.storage_state()
        raw = json.dumps(state)
        STATE_PATH.write_text(raw)
        if KV.r:
            await KV.r.set("shadowv1:state", raw)
    except Exception:
        pass


async def make_context():
    kwargs = dict(viewport=VIEW, user_agent=UA)
    saved = await KV.get("shadowv1:state") if KV.r else None
    if saved:
        STATE_PATH.write_text(saved)
        log("[store] restored saved DeepSeek login")
    if STATE_PATH.exists():
        kwargs["storage_state"] = str(STATE_PATH)
    elif os.getenv("DEEPSEEK_STATE"):
        kwargs["storage_state"] = json.loads(base64.b64decode(os.environ["DEEPSEEK_STATE"]))
    return await State.browser.new_context(**kwargs)


async def read_status(page):
    """Read login status from the DeepSeek page (page must be on SITE)."""
    try:
        info = await page.evaluate(JS_STATUS)
    except Exception:
        return
    State.status = {"connected": bool(info.get("connected")), "account": info.get("account")}
    State.status_at = time.time()


async def refresh_status():
    async with State.lock:
        page = State.page
        try:
            if not page.url.startswith(SITE):
                await page.goto(SITE, wait_until="domcontentloaded")
                await asyncio.sleep(1.5)
            await read_status(page)
        except Exception:
            pass


def _on_new_page(p):
    # popups (e.g. the Google sign-in window) become part of the remote view
    if State.login_active:
        State.login_pages.append(p)
        asyncio.create_task(_size_login_page(p))


async def _size_login_page(p):
    try:
        await p.set_viewport_size(LOGIN_VIEW)
    except Exception:
        pass


async def _route(route):
    """Skip images/fonts on the DeepSeek page: faster loads, less memory (but not while signing in)."""
    try:
        if not State.login_active and route.request.resource_type in ("image", "media", "font"):
            await route.abort()
        else:
            await route.continue_()
    except Exception:
        pass


async def background_loop():
    """Keeps the free instance awake and the login saved."""
    while True:
        await asyncio.sleep(540)
        if EXTERNAL_URL:
            try:
                await State.http.get(EXTERNAL_URL + "/health", timeout=20)
            except Exception:
                pass
        if not State.lock.locked():
            await save_state()


def _healthy() -> bool:
    try:
        return bool(State.browser and State.browser.is_connected() and State.page and not State.page.is_closed())
    except Exception:
        return False


def _is_dead(e: Exception) -> bool:
    m = str(e).lower()
    return any(k in m for k in ("has been closed", "target closed", "browser has been", "crashed",
                                "disconnected", "connection closed", "page closed"))


async def _launch():
    """(Re)start Chromium + the DeepSeek page. Login comes back from the saved state. Caller holds no lock needs."""
    for obj in (State.ctx, State.browser):
        if obj:
            try:
                await asyncio.wait_for(obj.close(), 6)
            except Exception:
                pass
    State.login_pages, State.login_active, State.warm = [], False, False
    State.browser = await State.pw.chromium.launch(headless=True, args=LAUNCH_ARGS)
    State.ctx = await make_context()
    State.ctx.on("page", _on_new_page)
    await State.ctx.route("**/*", _route)
    State.page = await State.ctx.new_page()
    try:
        await State.page.goto(SITE, wait_until="domcontentloaded", timeout=45000)
        await asyncio.sleep(1.5)
        await read_status(State.page)
        State.warm = bool(State.status.get("connected"))
    except Exception:
        pass


async def relaunch(reason=""):
    async with State.lock:
        if _healthy():
            return
        log("[browser] restarting:", reason[:120])
        await _launch()
        log("[browser] back. DeepSeek connected:", State.status.get("connected"))


async def watchdog():
    while True:
        await asyncio.sleep(15)
        if not _healthy() and not State.lock.locked():
            try:
                await relaunch("watchdog")
            except Exception as e:
                log("[browser] restart failed:", type(e).__name__, str(e)[:100])


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not API_KEYS:
        raise RuntimeError("Set the MY_API_KEY environment variable first.")
    State.http = httpx.AsyncClient(follow_redirects=True, timeout=30)
    await KV.start()
    State.pw = await async_playwright().start()
    await _launch()
    log("[boot] DeepSeek connected:", State.status.get("connected"))
    await load_tg_state()
    await load_cfg()
    start_telegram()
    State.tasks = [asyncio.create_task(background_loop()), asyncio.create_task(watchdog())]
    yield
    for t in State.tasks + [getattr(State, 'tg_task', None)]:
        if t:
            t.cancel()
    await save_state()
    await State.browser.close()
    await State.pw.stop()
    await State.http.aclose()


app = FastAPI(lifespan=lifespan)


def _clean_token(t: str) -> str:
    t = (t or "").strip().strip("\"'`").strip()
    while t.lower().startswith("bearer "):
        t = t[7:].strip().strip("\"'`").strip()
    return t


def _token_of(authorization=None, key=None, x_api_key=None):
    # accept "Bearer KEY", a bare "KEY", ?key=KEY and the x-api-key header
    return _clean_token(authorization) or _clean_token(key) or _clean_token(x_api_key)


def check_key(authorization=None, key=None, x_api_key=None):
    token = _token_of(authorization, key, x_api_key)
    ok = bool(token) and any(secrets.compare_digest(token.encode(), k.encode()) for k in API_KEYS)
    if not ok:
        shown = (token[:5] + "..." + token[-3:] + f" (len {len(token)})") if len(token) > 10 else repr(token)
        log("[auth] rejected key:", shown, "| expected len:", sorted({len(k) for k in API_KEYS}))
        raise HTTPException(401, "Invalid API key. Use the exact key from your Render MY_API_KEY setting.")


@app.middleware("http")
async def _auth_headers(request, call_next):
    """Cline/Continue/Azure-style clients put the key in different headers: fold them into Authorization."""
    hdrs = request.headers
    if not hdrs.get("authorization", "").strip():
        alt = hdrs.get("x-api-key") or hdrs.get("api-key") or hdrs.get("openai-api-key") or ""
        if alt.strip():
            request.scope["headers"] = [(k, v) for k, v in request.scope["headers"] if k != b"authorization"] + \
                                       [(b"authorization", ("Bearer " + alt.strip()).encode())]
    return await call_next(request)


async def guard(coro):
    try:
        return await coro
    except (IntegrationError, MediaError) as e:
        raise HTTPException(400, str(e))


def abs_url(path: str) -> str:
    return (EXTERNAL_URL or "") + path


# ---------------- status ----------------
@app.get("/status")
async def status(authorization: str | None = Header(default=None)):
    check_key(authorization)
    if not State.lock.locked() and time.time() - State.status_at > 20:
        await refresh_status()
    return {**State.status, "features": {
        "store": bool(KV.r), "telegram": bool(CFG.get("telegram_token")),
        "github": bool(CFG.get("github_token") or os.getenv("GITHUB_TOKEN")), "render": bool(CFG.get("render_key") or os.getenv("RENDER_API_KEY")),
        "image_key": bool(os.getenv("POLLINATIONS_KEY"))}}


# ---------------- chat core ----------------
async def set_toggle(page, labels, want: bool) -> dict:
    """Switch DeepThink / Search on or off in the DeepSeek page."""
    res = {"requested": want, "applied": False, "found": False}
    try:
        for _ in range(3):
            info = await page.evaluate(JS_TOGGLE, [labels, want])
            if not info.get("found"):
                return res
            res["found"] = True
            if bool(info["on"]) == want:
                res["applied"] = True
                return res
            await page.mouse.click(info["x"], info["y"])
            await asyncio.sleep(0.35)
    except Exception:
        pass
    return res


async def _send_and_wait(page, prompt: str, think: bool, search: bool, on_delta=None):
    """Type the prompt, press Enter, wait for the full answer. Returns (grab, thinking_count_before)."""
    g0 = await page.evaluate(JS_GRAB)
    before, before_think = g0["nAns"], g0["nThink"]
    done = asyncio.Event()

    def on_resp(resp):      # the answer arrives as one streamed network response; when it ends we are done
        try:
            if resp.request.method == "POST" and "/chat/completion" in resp.url:
                async def waiter():
                    try:
                        await resp.finished()
                    except Exception:
                        pass
                    done.set()
                asyncio.create_task(waiter())
        except Exception:
            pass

    page.on("response", on_resp)
    try:
        box = page.locator(SEL_INPUT).first
        await box.click()
        await box.fill(prompt)
        if len(prompt) > 1200:                     # long messages: let the page enable its send button first
            await asyncio.sleep(0.35)
        await page.keyboard.press("Enter")
        resend_at = time.time() + 2.5

        t0 = time.time()
        first_deadline = t0 + (360 if think else 180 if search else 90)
        last_text, changed_at, last_think = "", time.time(), ""
        poll = 0.15 if on_delta else 0.25
        while True:
            g = await page.evaluate(JS_GRAB)
            now = time.time()
            if on_delta and think and g["nThink"] > before_think and g["think_html"] != last_think:
                last_think = g["think_html"]
                await on_delta("think", html_to_md(last_think, partial=True))
            if not (g["nAns"] > before and g["ans_text"].strip()):
                if now > resend_at:                    # still no answer: was the message really sent?
                    resend_at = now + 4
                    try:
                        left = await page.evaluate("() => { const t = document.querySelector('textarea'); return t ? t.value.length : 0 }")
                        if left > 5:
                            log("[send] message still in the box, pressing Enter again")
                            await page.keyboard.press("Enter")
                    except Exception:
                        pass
                if now > first_deadline:
                    try:
                        seen = await page.evaluate("() => document.body.innerText.slice(-300)")
                    except Exception:
                        seen = "?"
                    log("[send] no reply. Page shows:", " ".join(str(seen).split())[:300])
                    raise HTTPException(504, "No reply from DeepSeek (it may be busy or showing an error). Try again.")
                await asyncio.sleep(poll)
                continue
            cur = g["ans_text"]
            if cur != last_text:
                last_text, changed_at = cur, now
                if on_delta:
                    await on_delta("answer", html_to_md(g["ans_html"], partial=True))
            need = 0.4 if done.is_set() else (3.0 if (think or search) else 2.0)
            if now - changed_at >= need or now > t0 + 600:
                return g, before_think
            await asyncio.sleep(poll)
    finally:
        try:
            page.remove_listener("response", on_resp)
        except Exception:
            pass


async def ask_deepseek(*a, **k) -> dict:
    """Ask DeepSeek. If the browser died (out of memory, crash, redeploy), restart it and retry once."""
    t0 = time.time()
    for attempt in (0, 1):
        try:
            r = await _ask_once(*a, **k)
            diag_add("deepseek", time.time() - t0, True, f"{len(r.get('reply', ''))} chars")
            return r
        except HTTPException as e:
            diag_add("deepseek", time.time() - t0, False, f"{e.status_code} {e.detail}")
            raise
        except Exception as e:
            if attempt == 0 and _is_dead(e):
                await relaunch(str(e))
                continue
            raise


async def _ask_once(message: str, chat_url: str | None = None, think: bool = False,
                    search: bool = False, system: bool = True, extra_system: str = "",
                    agent: bool = False, on_delta=None, stateless: bool = False) -> dict:
    async with State.lock:  # one message at a time
        if not _healthy():
            log("[browser] not healthy before request, restarting")
            await _launch()
        page = State.page
        new_chat = not (chat_url and chat_url.startswith(SITE))
        if new_chat:
            if not State.warm:
                await page.goto(SITE, wait_until="domcontentloaded")
        elif page.url != chat_url:
            await page.goto(chat_url, wait_until="domcontentloaded")
        State.warm = False
        State.last_stateless = stateless
        try:
            await page.wait_for_selector(SEL_INPUT, timeout=25000)
        except Exception:
            State.status = {"connected": False, "account": None}
            raise HTTPException(409, "Not logged in to DeepSeek. Connect your account first.")
        if "sign_in" in page.url:
            State.status = {"connected": False, "account": None}
            raise HTTPException(409, "Not logged in to DeepSeek. Connect your account first.")

        features = {
            "think": await set_toggle(page, THINK_LABELS, think),
            "search": await set_toggle(page, SEARCH_LABELS, search),
        }
        if new_chat and system:
            prompt = with_system(message, extra_system, agent)
        elif system and not agent:
            prompt = message + REMINDER
        else:
            prompt = message

        g, before_think = await _send_and_wait(page, prompt, think, search, on_delta)
        g_first = g
        reply = html_to_md(g["ans_html"]) or g["ans_text"].strip()
        if not on_delta and has_cjk_prose(reply):      # never Chinese: ask for a rewrite once
            try:
                g2, _ = await _send_and_wait(page, FIX_CHINESE, think, search)
                r2 = html_to_md(g2["ans_html"]) or g2["ans_text"].strip()
                if r2:
                    reply, g = r2, g2
            except Exception:
                pass
        reply = clean_reply(reply)
        thinking_html = clean_html(g_first["think_html"]) if g_first["nThink"] > before_think else ""
        result = {
            "reply": reply,
            "html": clean_html(g["ans_html"]),
            "thinking": clean_reply(html_to_md(thinking_html)) if thinking_html else "",
            "thinking_html": thinking_html,
            "chat_url": page.url,
            "features": features,
        }
        State.status["connected"] = True
    if time.time() - getattr(State, "saved_at", 0) > 120:      # saving is slow: do it in the background
        State.saved_at = time.time()
        asyncio.create_task(save_state())
    if stateless:
        asyncio.create_task(_prewarm())
    return result


async def _prewarm():
    """After a one-off request, get a fresh empty chat ready so the next request starts instantly."""
    try:
        async with State.lock:
            if not State.last_stateless or State.login_active:
                return
            await State.page.goto(SITE, wait_until="domcontentloaded")
            await State.page.wait_for_selector(SEL_INPUT, timeout=20000)
            State.warm = True
    except Exception:
        State.warm = False


async def respond_json(coro, shape):
    """Run a long job; if it takes >8s, send spaces as heartbeats so proxies don't time out."""
    task = asyncio.create_task(coro)
    done, _ = await asyncio.wait({task}, timeout=8)
    if done:
        return JSONResponse(shape(task.result()))   # raises HTTPException with the right status

    async def gen():
        yield b" "      # send the first byte now so proxies keep the connection open
        while True:
            d, _ = await asyncio.wait({task}, timeout=15)
            if d:
                break
            yield b" "
        try:
            yield json.dumps(shape(task.result())).encode()
        except HTTPException as e:
            yield json.dumps({"error": True, "status": e.status_code, "detail": e.detail}).encode()
        except Exception as e:
            yield json.dumps({"error": True, "status": 500, "detail": str(e)}).encode()

    return StreamingResponse(gen(), media_type="application/json")


# ---------------- images + pictures ----------------
IMG_CACHE: OrderedDict = OrderedDict()


async def make_image(prompt: str, w=1024, h=1024, seed=None, style=""):
    seed = seed or random.randint(1, 2_000_000)
    key = (prompt, w, h, seed, style)
    if key in IMG_CACHE:
        IMG_CACHE.move_to_end(key)
        data, ctype = IMG_CACHE[key]
    else:
        async with State.img_sem:
            data, ctype = await generate_image(State.http, prompt, w, h, seed, style)
        IMG_CACHE[key] = (data, ctype)
        while len(IMG_CACHE) > 6:
            IMG_CACHE.popitem(last=False)
    return data, ctype, img_path(API_KEYS[0], prompt, w, h, seed, style)


@app.get("/media/img")
async def media_img(prompt: str, w: int = 1024, h: int = 1024, seed: int = 0, style: str = "", sig: str = ""):
    if not secrets.compare_digest(sig, sign(API_KEYS[0], prompt, w, h, seed, style)):
        raise HTTPException(403, "Bad link")
    data, ctype, _ = await guard(make_image(prompt, w, h, seed, style))
    return Response(data, media_type=ctype, headers={"Cache-Control": "public, max-age=86400"})


def _pics_md(query, pics):
    return "\n\n".join(
        f"![{query}]({x['url']})" + (f"\n[Open source]({x['link']})" if x.get("link") else "") for x in pics)


def _pics_html(pics):
    return "".join(f'<p><img src="{x["url"]}" referrerpolicy="no-referrer" style="max-width:100%;border-radius:10px"></p>'
                   for x in pics)


async def smart_chat(message, chat_url=None, think=False, search=False, system=True, intent="auto"):
    """Chat, or (for 'draw ...' / 'find 5 anime pics') images and pictures."""
    kind, p = ("chat", {}) if intent == "chat" else detect_intent(message)
    base = {"thinking": "", "thinking_html": "", "chat_url": chat_url, "features": {}}
    if kind == "image":
        _, _, path = await guard(make_image(p["prompt"]))
        url = abs_url(path)
        return {**base, "type": "image", "reply": f"![{p['prompt'][:80]}]({url})", "images": [url],
                "html": f'<p><img src="{url}" style="max-width:100%;border-radius:10px"></p>'}
    if kind == "pins":
        pics = await guard(search_pictures(State.http, p["query"], p["amount"]))
        return {**base, "type": "pins", "reply": _pics_md(p["query"], pics), "html": _pics_html(pics),
                "images": [x["url"] for x in pics]}
    res = await ask_deepseek(message, chat_url, think, search, system)
    return {**res, "reply": clean_reply(res["reply"]), "type": "chat", "images": []}


class ChatIn(BaseModel):
    message: str
    chat_url: str | None = None
    think: bool = False
    search: bool = False
    system: bool = True
    intent: str = "auto"


@app.post("/chat")
async def chat(body: ChatIn, authorization: str | None = Header(default=None)):
    check_key(authorization)
    return await respond_json(
        smart_chat(body.message, body.chat_url, body.think, body.search, body.system, body.intent), lambda r: r)


# ---------------- OpenAI-compatible API (works with Cline, Continue, Roo, any bot) ----------------
MODELS = ["shadowv1", "shadowv1-think", "shadowv1-search", "shadowv1-think-search"]


class OAIMessage(BaseModel):
    role: str
    content: str | list | None = None


class OAIIn(BaseModel):
    model: str = "shadowv1"
    messages: list[OAIMessage]
    stream: bool = False
    think: bool | None = None
    search: bool | None = None
    model_config = {"extra": "allow"}


@app.get("/v1/models")
@app.get("/models")
async def list_models(authorization: str | None = Header(default=None)):
    check_key(authorization)
    return {"object": "list", "data": [
        {"id": m, "object": "model", "created": 1700000000, "owned_by": "vortex"} for m in MODELS]}


def _oai_payload(model, res, prompt_len):
    msg = {"role": "assistant", "content": clean_reply(res["reply"])}
    if res.get("thinking"):
        msg["reasoning_content"] = clean_reply(res["thinking"])
    pt, ct = prompt_len // 4, len(res["reply"]) // 4
    return {
        "id": f"chatcmpl-{secrets.token_hex(8)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": msg, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct},
        "chat_url": res["chat_url"],
    }


async def _oai_stream(start, model, prompt_len=0, include_usage=False):
    """Real streaming: text is sent while DeepSeek is still writing it."""
    cid = f"chatcmpl-{secrets.token_hex(8)}"
    q: asyncio.Queue = asyncio.Queue()

    async def on_delta(kind, text):
        await q.put((kind, text))

    def chunk(delta, finish=None):
        return "data: " + json.dumps({
            "id": cid, "object": "chat.completion.chunk", "created": int(time.time()),
            "model": model, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}) + "\n\n"

    task = asyncio.create_task(start(on_delta))
    sent = {"answer": "", "think": ""}
    field = {"answer": "content", "think": "reasoning_content"}
    yield chunk({"role": "assistant", "content": ""})

    def diff(kind, text, final=False):
        text = clean_reply(text)
        if not final:
            text = text[:-60] if len(text) > 60 else ""     # hold back the tail: it may still change
        s = sent[kind]
        if text.startswith(s):
            new = text[len(s):]
        elif final:
            i = 0
            while i < min(len(s), len(text)) and s[i] == text[i]:
                i += 1
            new = text[i:]
        else:
            return ""
        if new:
            sent[kind] = s + new if text.startswith(s) else text
        return new

    while not (task.done() and q.empty()):
        try:
            kind, text = await asyncio.wait_for(q.get(), timeout=10)
        except asyncio.TimeoutError:
            yield ": keep-alive\n\n"
            continue
        new = diff(kind, text)
        if new:
            yield chunk({field[kind]: new})
    try:
        res = task.result()
    except HTTPException as e:
        yield "data: " + json.dumps({"error": {"message": e.detail, "code": e.status_code}}) + "\n\n"
        yield "data: [DONE]\n\n"
        return
    except Exception as e:
        yield "data: " + json.dumps({"error": {"message": str(e), "code": 500}}) + "\n\n"
        yield "data: [DONE]\n\n"
        return
    for kind, text in (("think", res.get("thinking", "")), ("answer", res["reply"])):
        new = diff(kind, text, final=True)
        for i in range(0, len(new), 400):
            yield chunk({field[kind]: new[i:i + 400]})
    yield chunk({}, "stop")
    if include_usage:
        pt, ct = prompt_len // 4, len(res["reply"]) // 4
        yield "data: " + json.dumps({"id": cid, "object": "chat.completion.chunk", "created": int(time.time()),
                                     "model": model, "choices": [],
                                     "usage": {"prompt_tokens": pt, "completion_tokens": ct,
                                               "total_tokens": pt + ct}}) + "\n\n"
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def oai(body: OAIIn, authorization: str | None = Header(default=None)):
    check_key(authorization)
    think, search = flags_from_model(body.model, body.think, body.search)
    prompt, extra = build_convo(body.messages)
    agent = is_agent_request(extra, bool(getattr(body, "tools", None)))
    opts = getattr(body, "stream_options", None) or {}
    if body.stream:
        def start(on_delta):
            return ask_deepseek(prompt, None, think, search, True, extra, agent, on_delta, True)
        return StreamingResponse(
            _oai_stream(start, body.model, len(prompt), bool(opts.get("include_usage") if isinstance(opts, dict) else False)),
            media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
    return await respond_json(ask_deepseek(prompt, None, think, search, True, extra, agent, None, True),
                              lambda r: _oai_payload(body.model, r, len(prompt)))


class ImgIn(BaseModel):
    prompt: str
    n: int = 1
    size: str = "1024x1024"
    style: str = ""
    response_format: str = "url"
    model_config = {"extra": "allow"}


@app.post("/v1/images/generations")
@app.post("/images/generations")
async def oai_images(body: ImgIn, authorization: str | None = Header(default=None)):
    check_key(authorization)
    m = re.match(r"(\d+)x(\d+)", body.size or "")
    w, h = (int(m.group(1)), int(m.group(2))) if m else (1024, 1024)
    data = []
    for _ in range(max(1, min(body.n, 4))):
        img, _, path = await guard(make_image(body.prompt, w, h, None, body.style))
        data.append({"b64_json": base64.b64encode(img).decode()} if body.response_format == "b64_json"
                    else {"url": abs_url(path)})
    return {"created": int(time.time()), "data": data}


@app.get("/img/{prompt:path}")
async def img_get(prompt: str, w: int = 1024, h: int = 1024, seed: int = 0, style: str = "",
                  key: str | None = None, authorization: str | None = Header(default=None),
                  x_api_key: str | None = Header(default=None)):
    check_key(authorization, key, x_api_key)
    data, ctype, _ = await guard(make_image(prompt, w, h, seed or None, style))
    return Response(data, media_type=ctype)


@app.get("/pins/{query:path}")
async def pins_get(query: str, n: int = 5, key: str | None = None, authorization: str | None = Header(default=None),
                   x_api_key: str | None = Header(default=None)):
    check_key(authorization, key, x_api_key)
    pics = await guard(search_pictures(State.http, query, n))
    return {"query": query, "count": len(pics), "results": pics}


# ---------------- GitHub + Render ----------------
async def do_push(repo: str, md: str, message: str = "", branch: str | None = None) -> dict:
    files = extract_files(md)
    res = await gh().commit_files(repo, files, message or "Update by ShadowV1", branch)
    res["paths"] = [p for p, _ in files]
    return res


async def do_build(idea: str, private: bool = False, think: bool = False) -> dict:
    res = await ask_deepseek(idea + BUILD_SUFFIX, None, think, False, True)
    files = extract_files(res["reply"])
    if not files:
        return {"built": False, "reply": res["reply"]}
    name = f"{slugify(idea)}-{secrets.token_hex(2)}"
    repo = await gh().create_repo(name, private, f"{idea[:100]} (built by {BOT_NAME})")
    commit = await gh().commit_files(repo["name"], files, f"Initial commit by {BOT_NAME}", repo["default_branch"])
    out = {"built": True, "reply": res["reply"], "repo": repo["name"], "repo_url": repo["url"],
           "files": [p for p, _ in files], "commit": commit["url"], "site": None, "note": ""}
    paths = {p for p, _ in files}
    runtime = ("docker" if "Dockerfile" in paths else "node" if "package.json" in paths
               else "python" if "requirements.txt" in paths else "")
    if os.getenv("RENDER_API_KEY") and runtime:
        try:
            svc = await rn().create_web_service(name, repo["url"], await render_owner(), runtime,
                                                repo["default_branch"] or "main")
            out["site"] = svc
        except IntegrationError as e:
            out["note"] = f"Pushed to GitHub, but hosting on Render failed: {e}"
    elif not os.getenv("RENDER_API_KEY"):
        out["note"] = "Add RENDER_API_KEY to host it automatically."
    return out


class PushIn(BaseModel):
    repo: str
    message: str = ""
    markdown: str = ""
    branch: str | None = None


class NewRepoIn(BaseModel):
    name: str
    private: bool = True


class RunIn(BaseModel):
    repo: str
    workflow: str
    ref: str | None = None


class DeployIn(BaseModel):
    service: str = ""
    clear_cache: bool = False


class NewSiteIn(BaseModel):
    name: str
    repo_url: str
    runtime: str = "docker"
    branch: str = "main"


class BuildIn(BaseModel):
    idea: str
    private: bool = False
    think: bool = False


@app.get("/github/repos")
async def api_repos(authorization: str | None = Header(default=None)):
    check_key(authorization)
    return {"repos": await guard(gh().repos())}


@app.post("/github/newrepo")
async def api_newrepo(body: NewRepoIn, authorization: str | None = Header(default=None)):
    check_key(authorization)
    return await guard(gh().create_repo(body.name, body.private))


@app.post("/github/push")
async def api_push(body: PushIn, authorization: str | None = Header(default=None)):
    check_key(authorization)
    return await guard(do_push(body.repo, body.markdown, body.message, body.branch))


@app.get("/github/runs")
async def api_runs(repo: str, authorization: str | None = Header(default=None)):
    check_key(authorization)
    return {"runs": await guard(gh().runs(repo))}


@app.post("/github/run")
async def api_run(body: RunIn, authorization: str | None = Header(default=None)):
    check_key(authorization)
    return await guard(gh().dispatch(body.repo, body.workflow, body.ref))


@app.get("/render/services")
async def api_services(authorization: str | None = Header(default=None)):
    check_key(authorization)
    return {"services": await guard(rn().services())}


@app.post("/render/deploy")
async def api_deploy(body: DeployIn, authorization: str | None = Header(default=None)):
    check_key(authorization)
    svc = await guard(rn().find(body.service or os.getenv("RENDER_SERVICE_ID", "")))
    return {"service": svc["name"], **await guard(rn().deploy(svc["id"], body.clear_cache))}


@app.get("/render/deploys")
async def api_deploys(service: str, authorization: str | None = Header(default=None)):
    check_key(authorization)
    svc = await guard(rn().find(service))
    return {"service": svc["name"], "deploys": await guard(rn().deploys(svc["id"]))}


@app.get("/render/logs")
async def api_logs(service: str, authorization: str | None = Header(default=None)):
    check_key(authorization)
    svc = await guard(rn().find(service))
    return {"service": svc["name"], "logs": await guard(rn().logs(svc["id"], svc["owner"]))}


@app.post("/render/newsite")
async def api_newsite(body: NewSiteIn, authorization: str | None = Header(default=None)):
    check_key(authorization)
    owner = await guard(render_owner())
    return await guard(rn().create_web_service(body.name, body.repo_url, owner, body.runtime, body.branch))


@app.post("/build")
async def api_build(body: BuildIn, authorization: str | None = Header(default=None)):
    check_key(authorization)
    return await respond_json(guard(do_build(body.idea, body.private, body.think)), lambda r: r)


# ---------------- saved settings (tokens live in the Key Value store, never in the page) ----------------
CFG: dict = {"github_token": "", "render_key": "", "telegram_token": "", "tg_ids": [], "pair_code": "",
             "github_login": "", "tg_username": ""}
from collections import deque
DIAG: deque = deque(maxlen=30)
BOOT_AT = time.time()


def diag_add(kind, secs, ok, note=""):
    DIAG.append({"at": time.strftime("%H:%M:%S", time.gmtime()), "kind": kind, "secs": round(secs, 1), "ok": ok, "note": note[:120]})
    log(f"[diag] {kind} {secs:.1f}s ok={ok} {note[:100]}")


TG_OWNER = int(os.getenv("TELEGRAM_OWNER_ID", "8978770775"))   # always allowed
TG_ALLOWED: set = set()
PAIR_FAILS: dict = {}


async def load_cfg():
    try:
        raw = await KV.get("shadowv1:cfg")
        if raw:
            CFG.update(json.loads(raw))
    except Exception:
        pass
    CFG["telegram_token"] = CFG.get("telegram_token") or os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    TG_ALLOWED.clear()
    TG_ALLOWED.update(int(x) for x in CFG.get("tg_ids", []))
    TG_ALLOWED.update(int(x) for x in re.findall(r"-?\d+", os.getenv("TELEGRAM_ALLOWED_IDS", "")))
    TG_ALLOWED.add(TG_OWNER)


async def save_cfg():
    TG_ALLOWED.add(TG_OWNER)
    CFG["tg_ids"] = sorted(TG_ALLOWED)
    await KV.set("shadowv1:cfg", json.dumps(CFG))


def gh() -> GitHub:
    return GitHub(State.http, CFG.get("github_token") or None)


def rn() -> RenderAPI:
    return RenderAPI(State.http, CFG.get("render_key") or None)


async def render_owner() -> str:
    o = os.getenv("RENDER_OWNER_ID", "").strip()
    if o:
        return o
    svcs = await rn().services(5)
    if not svcs:
        raise IntegrationError("No Render services found to read the workspace from. Set RENDER_OWNER_ID.")
    return svcs[0]["owner"]


# ---------------- AI agent mode (GitHub, Render, terminal). Only here, never in plain chat ----------------
AGENT_SESS: dict = {}


async def agent_ask(message, url, first, think=False):
    return await ask_deepseek(message, url, think, False, True, AG.PROTOCOL if first else "", True)


async def run_agent(key: str, message: str, think: bool = False, emit=None) -> dict:
    sess = AGENT_SESS.setdefault(key, {"url": None, "pending": None, "lock": asyncio.Lock()})
    steps: list = []

    async def _emit(t):
        steps.append(t)
        if emit:
            await emit(t)
    async with sess["lock"]:
        res = await AG.drive(sess, message, lambda m, u, f: agent_ask(m, u, f, think),
                             AG.Toolbox(gh, rn), _emit)
    res["steps"] = steps
    res["reply"] = clean_reply(res.get("reply", ""))
    return res


class AgentIn(BaseModel):
    message: str
    session: str = "web"
    think: bool = False


@app.post("/agent")
async def api_agent(body: AgentIn, authorization: str | None = Header(default=None)):
    check_key(authorization)
    return await respond_json(run_agent("web:" + body.session[:40], body.message, body.think), lambda r: r)


@app.post("/agent/reset")
async def api_agent_reset(session: str = "web", authorization: str | None = Header(default=None)):
    check_key(authorization)
    AGENT_SESS.pop("web:" + session[:40], None)
    return {"ok": True}


class CfgIn(BaseModel):
    github_token: str | None = None
    render_key: str | None = None
    telegram_token: str | None = None


def _cfg_view() -> dict:
    return {"store": bool(KV.r),
            "github": {"set": bool(CFG.get("github_token") or os.getenv("GITHUB_TOKEN")), "login": CFG.get("github_login", "")},
            "render": {"set": bool(CFG.get("render_key") or os.getenv("RENDER_API_KEY"))},
            "telegram": {"set": bool(CFG.get("telegram_token")), "username": CFG.get("tg_username", ""),
                         "paired": len(TG_ALLOWED), "pair_code": CFG.get("pair_code", "")}}


@app.get("/diag")
async def get_diag(authorization: str | None = Header(default=None)):
    check_key(authorization)
    return {"browser_ok": _healthy(), "deepseek_connected": State.status.get("connected"), "busy": State.lock.locked(),
            "uptime_min": int((time.time() - BOOT_AT) / 60), "recent": list(DIAG)}


@app.get("/config")
async def get_config(authorization: str | None = Header(default=None)):
    check_key(authorization)
    return _cfg_view()


@app.post("/config")
async def set_config(body: CfgIn, authorization: str | None = Header(default=None)):
    check_key(authorization)
    if body.github_token is not None:
        tok = body.github_token.strip()
        if tok:
            CFG["github_login"] = await guard(GitHub(State.http, tok).me())
        else:
            CFG["github_login"] = ""
        CFG["github_token"] = tok
    if body.render_key is not None:
        CFG["render_key"] = body.render_key.strip()
        if CFG["render_key"]:
            await guard(RenderAPI(State.http, CFG["render_key"]).services(1))
    if body.telegram_token is not None:
        tok = body.telegram_token.strip()
        if tok:
            r = await State.http.post(f"https://api.telegram.org/bot{tok}/getMe", timeout=20)
            j = r.json()
            if not j.get("ok"):
                raise HTTPException(400, "Telegram rejected that token. Copy it again from @BotFather.")
            if tok != CFG.get("telegram_token"):
                TG_ALLOWED.clear()
                TG_ALLOWED.add(TG_OWNER)
            CFG["tg_username"] = (j.get("result") or {}).get("username", "")
            CFG["pair_code"] = secrets.token_hex(4)
        else:
            CFG["tg_username"], CFG["pair_code"] = "", ""
            TG_ALLOWED.clear()
        CFG["telegram_token"] = tok
        start_telegram()
    await save_cfg()
    return _cfg_view()


@app.post("/config/unpair")
async def unpair(authorization: str | None = Header(default=None)):
    check_key(authorization)
    TG_ALLOWED.clear()
    TG_ALLOWED.add(TG_OWNER)
    CFG["pair_code"] = secrets.token_hex(4) if CFG.get("telegram_token") else ""
    await save_cfg()
    return _cfg_view()


# ---------------- Telegram bot ----------------
TG_STATE: dict = {}


async def load_tg_state():
    try:
        raw = await KV.get("shadowv1:tg")
        if raw:
            TG_STATE.update({int(k): v for k, v in json.loads(raw).items()})
    except Exception:
        pass


async def tg_send(call, chat_id, md):
    for chunk in split_md(md or "(empty reply)"):
        r = await call("sendMessage", chat_id=chat_id, text=md_to_tg(chunk), parse_mode="HTML",
                       disable_web_page_preview=False)
        if not r.get("ok"):
            for i in range(0, len(chunk), 4000):
                await call("sendMessage", chat_id=chat_id, text=chunk[i:i + 4000])


async def handle_tg(call, allowed, upd):
    msg = upd.get("message") or {}
    text = (msg.get("text") or "").strip()
    chat_id = (msg.get("chat") or {}).get("id")
    uid = (msg.get("from") or {}).get("id")
    if not text or chat_id is None:
        return
    parts = text.split()
    if uid not in allowed:
        if parts[0].split("@")[0].lower() == "/pair" and len(parts) > 1 and CFG.get("pair_code"):
            now = time.time()
            fails = [t for t in PAIR_FAILS.get(uid, []) if now - t < 600]
            if len(fails) >= 5:
                await call("sendMessage", chat_id=chat_id, text="Too many tries. Wait 10 minutes.")
            elif secrets.compare_digest(parts[1].strip().lower(), CFG["pair_code"]):
                allowed.add(uid)
                CFG["pair_code"] = secrets.token_hex(4)       # one-time code
                await save_cfg()
                await call("sendMessage", chat_id=chat_id,
                           text=f"Paired. Welcome to {BOT_NAME} by {BOT_MAKER}. Send /help.")
            else:
                PAIR_FAILS[uid] = fails + [now]
                await call("sendMessage", chat_id=chat_id, text="Wrong code.")
        else:
            await call("sendMessage", chat_id=chat_id,
                       text="Private bot. Open your ShadowV1 website, go to the Agent page, and send me: /pair CODE")
        return
    st = TG_STATE.setdefault(chat_id, {"url": None, "think": False, "search": False, "mode": "auto"})
    if not st.get("mode_set") and st.get("mode") not in ("chat", "agent", "auto"):
        st["mode"] = "auto"
    if not st.get("mode_set"):
        st["mode"] = "auto"
    cmd = parts[0].split("@")[0].lower() if text.startswith("/") else ""
    arg = parts[1].lower() if len(parts) > 1 else ""
    if cmd in ("/start", "/help"):
        await call("sendMessage", chat_id=chat_id, text=(
            f"{BOT_NAME} by {BOT_MAKER}.\n"
            "Just talk. For real work (GitHub, builds, deploys, APKs, debugging) I switch to agent mode myself.\n"
            "Deletes and merges always ask you to reply YES first.\n"
            "/mode auto|agent|chat   /new   /think on|off   /search on|off\n"
            "Say 'draw a ...' for an image or 'send 5 anime pics from pinterest' for photos."))
        return
    if cmd == "/diag":
        rows = "\n".join(f"{d['at']} {d['kind']} {d['secs']}s {'ok' if d['ok'] else 'FAIL'} {d['note']}" for d in list(DIAG)[-8:])
        await call("sendMessage", chat_id=chat_id, text=(
            f"Browser: {'ok' if _healthy() else 'DOWN'} | DeepSeek login: {State.status.get('connected')} | "
            f"busy: {State.lock.locked()} | up {int((time.time() - BOOT_AT) / 60)} min\n{rows or 'no requests yet'}"))
        return
    if cmd == "/mode":
        if arg in ("auto", "chat", "agent"):
            st["mode"], st["mode_set"] = arg, True
        await call("sendMessage", chat_id=chat_id, text=f"Mode: {st['mode']}")
        return
    if cmd == "/new":
        st["url"] = None
        AGENT_SESS.pop(f"tg:{chat_id}", None)
        await call("sendMessage", chat_id=chat_id, text="New chat started.")
        return
    if cmd in ("/think", "/search"):
        k = cmd[1:]
        st[k] = (arg == "on") if arg in ("on", "off") else (not st[k])
        await call("sendMessage", chat_id=chat_id, text=f"{k.capitalize()}: {'ON' if st[k] else 'OFF'}")
        return
    if cmd:
        text = text.lstrip("/")
    await tg_work(call, chat_id, msg.get("message_id"), text, st)
    await KV.set("shadowv1:tg", json.dumps(TG_STATE))


AGENT_HINT = re.compile(
    r"\b(github|git hub|repo|repository|commit|push|pull request|workflow|actions?|deploy|render|apk|build|"
    r"terminal|shell|install|debug|branch|issue|release|npm|pip|docker|scaffold|clone|merge|delete (the )?(repo|file|branch)|"
    r"(create|make|write|code)\s+(me\s+)?(a|an|the)?\s*(app|bot|website|site|api|project|script|game|tool))\b", re.I)
SPIN = ["◐", "◓", "◑", "◒"]


async def tg_react(call, chat_id, mid, emoji):
    if mid:
        await call("setMessageReaction", chat_id=chat_id, message_id=mid, reaction=[{"type": "emoji", "emoji": emoji}])


async def tg_photos(call, chat_id, pics, caption=""):
    """Send real photos as one album. Telegram can't fetch Pinterest links itself, so download in parallel and upload."""
    async def grab(x):
        try:
            return await asyncio.wait_for(fetch_bytes(State.http, x["url"]), 25)
        except Exception:
            return None
    got = [(x, d) for x, d in zip(pics[:10], await asyncio.gather(*[grab(x) for x in pics[:10]])) if d]
    if len(got) > 1:
        files, media = {}, []
        for i, (x, (data, ctype)) in enumerate(got):
            files[f"p{i}"] = (f"pic{i}.jpg", data, ctype)
            media.append({"type": "photo", "media": f"attach://p{i}", **({"caption": caption[:200]} if i == 0 else {})})
        r = await call("sendMediaGroup", chat_id=chat_id, media=media, files=files)
        if r.get("ok"):
            return len(got)
    sent = 0
    for x, (data, ctype) in got:
        r = await call("sendPhoto", chat_id=chat_id, caption=(caption if not sent else "")[:200],
                       files={"photo": ("pic.jpg", data, ctype)})
        sent += 1 if r.get("ok") else 0
    if not sent:                       # last resort: let Telegram try the links itself
        urls = [x["url"] for x in pics][:10]
        if len(urls) > 1:
            r = await call("sendMediaGroup", chat_id=chat_id,
                           media=[{"type": "photo", "media": u} for u in urls])
        else:
            r = await call("sendPhoto", chat_id=chat_id, photo=urls[0]) if urls else {}
        sent = len(urls) if r.get("ok") else 0
    return sent


async def tg_work(call, chat_id, mid, text, st):
    kind, p = detect_intent(text)
    sess = AGENT_SESS.get(f"tg:{chat_id}")
    pending = bool(sess and sess.get("pending"))
    use_agent = pending or (kind == "chat" and (st["mode"] == "agent" or
                                                (st["mode"] == "auto" and bool(AGENT_HINT.search(text)))))
    if st["mode"] == "chat" and not pending:
        use_agent = False
    await tg_react(call, chat_id, mid, "👀")
    if State.lock.locked():
        await call("sendMessage", chat_id=chat_id, text="⏳ Another request is running (website or another chat). "
                                                         "Yours is next.")
    action = "upload_photo" if kind in ("image", "pins") else "typing"
    typing = asyncio.create_task(_typing(call, chat_id, action))
    status = {"id": None, "text": "Thinking"}
    anim = None
    if use_agent or kind != "chat" or st["think"]:
        r = await call("sendMessage", chat_id=chat_id, text=f"{SPIN[0]} {status['text']}")
        status["id"] = (r.get("result") or {}).get("message_id")

        async def animate():
            i = 0
            while True:
                await asyncio.sleep(3)
                i += 1
                await call("editMessageText", chat_id=chat_id, message_id=status["id"],
                           text=f"{SPIN[i % 4]} {status['text']}{'.' * (i % 4)}")
        if status["id"]:
            anim = asyncio.create_task(animate())
    await tg_react(call, chat_id, mid, "👨‍💻" if use_agent else "🤩" if kind == "image" else "🤔")
    ok, note, t0 = True, "", time.time()
    try:
        async def work():
            if use_agent:
                async def emit(t):
                    status["text"] = "🔧 " + t[:120]
                res = await run_agent(f"tg:{chat_id}", text, st["think"], emit)
                await tg_send(call, chat_id, res["reply"])
            elif kind == "image":
                status["text"] = "Painting your image"
                img, ctype, _ = await guard(make_image(p["prompt"]))
                r = await call("sendPhoto", chat_id=chat_id, caption=p["prompt"][:200],
                               files={"photo": ("image.jpg", img, ctype)})
                if not r.get("ok"):
                    await tg_send(call, chat_id, "Could not send the image. Try again.")
            elif kind == "pins":
                status["text"] = f"Finding {p['amount']} pictures"
                pics = await guard(search_pictures(State.http, p["query"], p["amount"]))
                n = await tg_photos(call, chat_id, pics, p["query"])
                if not n:
                    await tg_send(call, chat_id, "\n".join(x["url"] for x in pics) or "No pictures found.")
            else:
                res = await smart_chat(text, st["url"], st["think"], st["search"], True, "chat")
                st["url"] = res["chat_url"]
                await tg_send(call, chat_id, res["reply"])
        await asyncio.wait_for(work(), 600 if use_agent else 240)
    except asyncio.TimeoutError:
        ok = False
        await call("sendMessage", chat_id=chat_id, text="That took too long. Try again, or send /new.")
    except HTTPException as e:
        ok, note = False, f"{e.status_code} {e.detail}"
        await call("sendMessage", chat_id=chat_id, text=f"Error: {e.detail}")
    except Exception as e:
        ok, note = False, f"{type(e).__name__} {e}"
        log("[telegram] work error:", type(e).__name__, str(e)[:200])
        await call("sendMessage", chat_id=chat_id, text=f"Error: {e}")
    finally:
        typing.cancel()
        if anim:
            anim.cancel()
        if status["id"]:
            await call("deleteMessage", chat_id=chat_id, message_id=status["id"])
    diag_add("tg-" + ("agent" if use_agent else kind), time.time() - t0, ok, note or text[:40])
    await tg_react(call, chat_id, mid, "🔥" if ok else "😢")


async def _typing(call, chat_id, action="typing"):
    while True:
        await call("sendChatAction", chat_id=chat_id, action=action)
        await asyncio.sleep(4)


def start_telegram():
    t = getattr(State, "tg_task", None)
    if t:
        t.cancel()
    State.tg_task = asyncio.create_task(telegram_loop())


async def telegram_loop():
    token = (CFG.get("telegram_token") or "").strip()
    if not token:
        log("[telegram] off (no token yet; add it on the Agent page)")
        return

    async def call(method, **kw):
        try:
            files = kw.pop("files", None)
            url = f"https://api.telegram.org/bot{token}/{method}"
            if files:
                r = await State.http.post(url, data={k: (json.dumps(v) if isinstance(v, (list, dict)) else str(v))
                                                     for k, v in kw.items()}, files=files, timeout=90)
            else:
                r = await State.http.post(url, json=kw, timeout=50)
            j = r.json()
            if not j.get("ok"):
                log(f"[telegram] {method}: {j.get('description')}")
            return j
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log(f"[telegram] {method} error: {type(e).__name__}")
            return {}

    await call("deleteWebhook")
    me = await call("getMe")
    log("[telegram] bot ready:", (me.get("result") or {}).get("username"), "paired ids:", len(TG_ALLOWED))
    offset = 0
    while True:
        data = await call("getUpdates", offset=offset, timeout=30, allowed_updates=["message"])
        if not data.get("ok"):
            await asyncio.sleep(5)
            continue
        for u in data.get("result", []):
            offset = u["update_id"] + 1
            asyncio.create_task(handle_tg(call, TG_ALLOWED, u))


# ---------------- connect via your own Chrome ----------------
@app.post("/connect/token")
async def connect_token(authorization: str | None = Header(default=None)):
    check_key(authorization)
    now = time.time()
    State.codes = {k: v for k, v in State.codes.items() if v > now}
    code = secrets.token_urlsafe(9)
    State.codes[code] = now + 600
    return {"code": code, "expires_in": 600}


class ImportIn(BaseModel):
    code: str = ""
    data: str


@app.post("/session/import")
async def session_import(body: ImportIn, authorization: str | None = Header(default=None)):
    if authorization:            # reusable bookmark: authorised with your API key
        check_key(authorization)
    else:                        # one-time code
        exp = State.codes.get(body.code, 0)
        if not exp or exp < time.time():
            raise HTTPException(403, "Code expired. Generate a new one on the Connect page.")
    try:
        raw = base64.b64decode(body.data + "=" * (-len(body.data) % 4))
        if len(raw) > 600_000:
            raise ValueError("too big")
        d = json.loads(raw)
        ls = {str(k): str(v) for k, v in d.get("ls", {}).items() if len(str(k)) < 200}
        ck = str(d.get("ck", ""))
    except Exception:
        raise HTTPException(400, "Could not read the data. Run it again on chat.deepseek.com.")
    if "userToken" not in ls:
        raise HTTPException(
            400, "No DeepSeek login found in that browser. Sign in on chat.deepseek.com first, then run it again.")

    cookies = []
    for part in ck.split(";"):
        if "=" in part:
            n, v = part.strip().split("=", 1)
            if n:
                cookies.append({"name": n, "value": v, "url": SITE})

    async with State.lock:
        if cookies:
            try:
                await State.ctx.add_cookies(cookies)
            except Exception:
                pass
        page = State.page
        await page.goto(SITE, wait_until="domcontentloaded")
        await page.evaluate(
            "items => { for (const [k, v] of Object.entries(items)) localStorage.setItem(k, v); }", ls)
        await page.reload(wait_until="domcontentloaded")
        await asyncio.sleep(2)
        await read_status(page)
    await save_state()
    State.codes.pop(body.code, None)
    if not State.status.get("connected"):
        raise HTTPException(400, "DeepSeek did not accept that login. Try the backup sign-in (/remote).")
    return State.status


# ---------------- remote login view (backup) ----------------
def active_login_page():
    pages = [p for p in State.login_pages if not p.is_closed()]
    return pages[-1] if pages else None


def need_page():
    p = active_login_page()
    if p is None:
        raise HTTPException(409, "Press Start first.")
    return p


@app.post("/remote/open")
async def remote_open(authorization: str | None = Header(default=None)):
    check_key(authorization)
    State.login_active = True
    if active_login_page() is None:
        p = await State.ctx.new_page()
        State.login_pages = [p]
        await _size_login_page(p)
    await State.login_pages[0].goto(SITE, wait_until="domcontentloaded")
    return {"ok": True}


@app.get("/remote/shot")
async def remote_shot(authorization: str | None = Header(default=None)):
    check_key(authorization)
    p = need_page()
    img = await p.screenshot(type="jpeg", quality=72)
    return Response(img, media_type="image/jpeg")


class ClickIn(BaseModel):
    x: float
    y: float
    hold_ms: float = 0


class Pt(BaseModel):
    x: float
    y: float
    t: float  # ms since touch start


class DragIn(BaseModel):
    points: list[Pt]


class ScrollIn(BaseModel):
    dy: float


class TypeIn(BaseModel):
    text: str


class KeyIn(BaseModel):
    key: str


@app.post("/remote/click")
async def remote_click(body: ClickIn, authorization: str | None = Header(default=None)):
    check_key(authorization)
    p = need_page()
    await p.mouse.move(body.x, body.y, steps=6)   # hover first, like a real mouse
    await asyncio.sleep(0.05)
    await p.mouse.down()
    await asyncio.sleep(min(max(body.hold_ms, 0), 3000) / 1000 or 0.08)
    await p.mouse.up()
    return {"ok": True}


@app.post("/remote/drag")
async def remote_drag(body: DragIn, authorization: str | None = Header(default=None)):
    check_key(authorization)
    p = need_page()
    pts = body.points[:200]
    if len(pts) < 2:
        raise HTTPException(400, "Not enough points")
    await p.mouse.move(pts[0].x, pts[0].y, steps=4)
    await asyncio.sleep(0.05)
    await p.mouse.down()
    prev = pts[0].t
    for pt in pts[1:]:
        await asyncio.sleep(min(max(pt.t - prev, 0), 120) / 1000)
        await p.mouse.move(pt.x, pt.y)
        prev = pt.t
    await asyncio.sleep(0.05)
    await p.mouse.up()
    return {"ok": True}


@app.post("/remote/scroll")
async def remote_scroll(body: ScrollIn, authorization: str | None = Header(default=None)):
    check_key(authorization)
    p = need_page()
    await p.mouse.move(LOGIN_VIEW["width"] / 2, LOGIN_VIEW["height"] / 2)
    await p.mouse.wheel(0, max(-1500, min(1500, body.dy)))
    return {"ok": True}


@app.post("/remote/type")
async def remote_type(body: TypeIn, authorization: str | None = Header(default=None)):
    check_key(authorization)
    p = need_page()
    await p.keyboard.type(body.text, delay=25)
    return {"ok": True}


@app.post("/remote/key")
async def remote_key(body: KeyIn, authorization: str | None = Header(default=None)):
    check_key(authorization)
    if body.key not in ALLOWED_KEYS:
        raise HTTPException(400, "Key not allowed")
    p = need_page()
    await p.keyboard.press(body.key)
    return {"ok": True}


@app.post("/remote/finish")
async def remote_finish(authorization: str | None = Header(default=None)):
    check_key(authorization)
    await save_state()
    State.login_active = False
    for p in State.login_pages:
        if not p.is_closed():
            await p.close()
    State.login_pages = []
    # reload the chat page so it picks up the new login, then read who is connected
    async with State.lock:
        try:
            await State.page.goto(SITE, wait_until="domcontentloaded")
            await asyncio.sleep(2)
            await read_status(State.page)
        except Exception:
            pass
    await save_state()
    return State.status


# ---------------- pages ----------------
@app.api_route("/", methods=["GET", "HEAD"])
async def index():
    return FileResponse(HERE / "index.html")


@app.get("/remote")
async def remote_page():
    return FileResponse(HERE / "remote.html")


@app.get("/connect")
async def connect_page():
    return FileResponse(HERE / "connect.html")


@app.get("/guide")
async def guide_page():
    return FileResponse(HERE / "guide.html")


@app.get("/agent")
async def agent_page():
    return FileResponse(HERE / "agent.html")


@app.get("/import")
async def import_page():
    return FileResponse(HERE / "import.html")


@app.api_route("/health", methods=["GET", "HEAD"])
async def health():
    return {"ok": True}


# ---------------- prompt URL:  https://HOST/<your prompt>?key=KEY  ----------------
_RESERVED = ("favicon.ico", "robots.txt", "apple-touch-icon.png", "sitemap.xml")


@app.get("/{prompt:path}")
async def prompt_api(prompt: str, key: str | None = None, think: bool = False, search: bool = False,
                     format: str = "text", chat: str | None = None,
                     authorization: str | None = Header(default=None),
                     x_api_key: str | None = Header(default=None)):
    if not prompt.strip() or prompt in _RESERVED or prompt.split("/")[0] in (
            "v1", "github", "render", "remote", "media", "docs", "openapi.json"):
        raise HTTPException(404, "Not found")
    try:
        check_key(authorization, key, x_api_key)
    except HTTPException:
        raise HTTPException(401, "Add your key: /your prompt?key=YOUR_KEY")
    kind, p = detect_intent(prompt)
    if kind == "image":
        data, ctype, _ = await guard(make_image(p["prompt"]))
        return Response(data, media_type=ctype)
    if kind == "pins":
        pics = await guard(search_pictures(State.http, p["query"], p["amount"]))
        if format == "json":
            return {"query": p["query"], "results": pics}
        return Response("\n".join(x["url"] for x in pics), media_type="text/plain; charset=utf-8")
    res = await ask_deepseek(prompt, chat, think, search, True, "", False, None, chat is None)
    if format == "json":
        return {**{k: res[k] for k in ("thinking", "chat_url", "features")}, "reply": clean_reply(res["reply"])}
    return Response(clean_reply(res["reply"]), media_type="text/plain; charset=utf-8",
                    headers={"X-Chat-Url": res["chat_url"]})


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    host = "0.0.0.0" if os.getenv("PORT") else "127.0.0.1"
    uvicorn.run(app, host=host, port=port)
