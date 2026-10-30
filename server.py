"""
ShadowV1 provider API: use your DeepSeek web chat as an OpenAI-compatible provider
(VS Code Cline/Roo/Continue, OpenCode, AntiGem custom provider, any app).

  POST /v1/chat/completions   chat (streaming works), with native tool calling for coding agents
  GET  /v1/models             model list (shadowv1, shadowv1-think, shadowv1-search, shadowv1-think-search)
  /                           info page: base URL, key, test button     /connect /remote  sign-in pages

Env vars: MY_API_KEY (required, comma = several keys), REDIS_URL (keeps your DeepSeek login across restarts)
The full bot (Telegram, agent, GitHub, images) lives on the git branch `full-bot-backup`.
"""
import asyncio
import base64
import json
import os
import re
import secrets
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Header, HTTPException, Response
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel
from playwright.async_api import async_playwright
import httpx

import toolcalls as TC
from shadow import (BOT_NAME, BOT_MAKER, JS_GRAB, JS_TOGGLE, REMINDER, SEARCH_LABELS, THINK_LABELS,
                    build_convo, clean_html, clean_reply, flags_from_model, has_cjk_prose, html_to_md,
                    is_agent_request, with_system)

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


DIAG: deque = deque(maxlen=30)
BOOT_AT = time.time()


def diag_add(kind, secs, ok, note=""):
    DIAG.append({"at": time.strftime("%H:%M:%S", time.gmtime()), "kind": kind, "secs": round(secs, 1),
                 "ok": ok, "note": note[:120]})
    log(f"[diag] {kind} {secs:.1f}s ok={ok} {note[:100]}")


class State:
    pw = None
    browser = None
    ctx = None
    page = None
    http = None
    login_pages: list = []
    login_active = False
    lock = asyncio.Lock()
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
    State.tasks = [asyncio.create_task(background_loop()), asyncio.create_task(watchdog())]
    yield
    for t in State.tasks:
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


# ---------------- status ----------------
@app.get("/status")
async def status(authorization: str | None = Header(default=None), x_api_key: str | None = Header(default=None)):
    check_key(authorization, None, x_api_key)
    if not State.lock.locked() and time.time() - State.status_at > 20:
        await refresh_status()
    return {**State.status, "store": bool(KV.r), "browser_ok": _healthy()}


@app.get("/diag")
async def get_diag(authorization: str | None = Header(default=None)):
    check_key(authorization)
    return {"browser_ok": _healthy(), "deepseek_connected": State.status.get("connected"), "busy": State.lock.locked(),
            "uptime_min": int((time.time() - BOOT_AT) / 60), "recent": list(DIAG)}


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


# ---------------- OpenAI-compatible API (works with Cline, Continue, Roo, any bot) ----------------
MODELS = ["shadowv1", "shadowv1-think", "shadowv1-search", "shadowv1-think-search"]


class OAIMessage(BaseModel):
    role: str
    content: str | list | None = None
    tool_calls: list | None = None
    tool_call_id: str | None = None
    name: str | None = None
    model_config = {"extra": "allow"}


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


def _tool_payload(model, res, prompt_len, names):
    text, calls = TC.parse_reply(res["reply"], names)
    msg = {"role": "assistant", "content": clean_reply(text) if text else None}
    if calls:
        msg["tool_calls"] = calls
    pt, ct = prompt_len // 4, len(res["reply"]) // 4
    return {"id": f"chatcmpl-{secrets.token_hex(8)}", "object": "chat.completion", "created": int(time.time()),
            "model": model, "choices": [{"index": 0, "message": msg, "finish_reason": "tool_calls" if calls else "stop"}],
            "usage": {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct}}


async def _tool_stream(model, task, prompt_len, names):
    """Tool mode can't stream text early (the reply may be a tool call), so keep the line alive, then send it."""
    cid = f"chatcmpl-{secrets.token_hex(8)}"

    def chunk(delta, finish=None):
        return "data: " + json.dumps({"id": cid, "object": "chat.completion.chunk", "created": int(time.time()),
                                      "model": model,
                                      "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}) + "\n\n"
    yield chunk({"role": "assistant", "content": ""})
    while True:
        done, _ = await asyncio.wait({task}, timeout=10)
        if done:
            break
        yield ": keep-alive\n\n"
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
    p = _tool_payload(model, res, prompt_len, names)["choices"][0]
    text = p["message"].get("content") or ""
    for i in range(0, len(text), 400):
        yield chunk({"content": text[i:i + 400]})
    for i, c in enumerate(p["message"].get("tool_calls") or []):
        yield chunk({"tool_calls": [{"index": i, "id": c["id"], "type": "function",
                                     "function": {"name": c["function"]["name"], "arguments": ""}}]})
        args = c["function"]["arguments"]
        for k in range(0, len(args), 400):
            yield chunk({"tool_calls": [{"index": i, "function": {"arguments": args[k:k + 400]}}]})
    yield chunk({}, p["finish_reason"])
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def oai(body: OAIIn, authorization: str | None = Header(default=None),
              x_api_key: str | None = Header(default=None)):
    check_key(authorization, None, x_api_key)
    think, search = flags_from_model(body.model, body.think, body.search)
    tools = [t for t in (getattr(body, "tools", None) or []) if isinstance(t, dict)]
    names = TC.tool_names(tools)
    if names and getattr(body, "tool_choice", None) != "none":          # native tool calling (OpenCode, AntiGem...)
        instr, convo = TC.build_prompt(body.messages, tools)
        plen = len(instr) + len(convo)
        coro = ask_deepseek(convo, None, think, search, True, instr, True, None, True)
        if body.stream:
            return StreamingResponse(_tool_stream(body.model, asyncio.create_task(coro), plen, names),
                                     media_type="text/event-stream",
                                     headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
        return await respond_json(coro, lambda r: _tool_payload(body.model, r, plen, names))
    prompt, extra = build_convo(body.messages)
    agent = is_agent_request(extra, False)
    opts = getattr(body, "stream_options", None) or {}
    if body.stream:
        def start(on_delta):
            return ask_deepseek(prompt, None, think, search, True, extra, agent, on_delta, True)
        return StreamingResponse(
            _oai_stream(start, body.model, len(prompt), bool(opts.get("include_usage") if isinstance(opts, dict) else False)),
            media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
    return await respond_json(ask_deepseek(prompt, None, think, search, True, extra, agent, None, True),
                              lambda r: _oai_payload(body.model, r, len(prompt)))


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
    return FileResponse(HERE / "index.html")


@app.get("/import")
async def import_page():
    return FileResponse(HERE / "import.html")


@app.api_route("/health", methods=["GET", "HEAD"])
async def health():
    return {"ok": True}


