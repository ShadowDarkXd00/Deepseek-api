"""
Unofficial DeepSeek web-chat -> your own website + API key.

  /          chat website (new chats, old chats, code blocks)
  /guide     how to use the API (curl, Python, Telegram, any bot)
  /connect   easiest login: sign in with YOUR Chrome (device Google accounts)
  /remote    backup login: live browser view with touch / drag / hold
  /chat      POST  {"message": "...", "chat_url": null}   (Bearer key)
  /v1/chat/completions   OpenAI-style endpoint (Bearer key)

Env vars:
  MY_API_KEY  (required)  your own key / password for everything
  DATA_DIR    (optional)  where the login session is saved (default ./data)
"""
import asyncio
import base64
import json
import os
import re
import secrets
import time
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Header, HTTPException, Response
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel
from playwright.async_api import async_playwright
import httpx

from shadow import (BOT_NAME, BOT_MAKER, JS_GRAB, JS_TOGGLE, SEARCH_LABELS, THINK_LABELS,
                    build_convo, flags_from_model, html_to_md, md_to_tg, split_md, with_system)

# ---------- CONFIG ----------
API_KEYS = [k.strip() for k in os.getenv("MY_API_KEY", "").split(",") if k.strip()]  # comma = several keys
DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
STATE_PATH = DATA_DIR / "state.json"
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
# ----------------------------


class State:
    pw = None
    browser = None
    ctx = None
    page = None
    login_pages: list = []
    login_active = False
    lock = asyncio.Lock()
    codes: dict = {}                      # one-time /connect codes -> expiry
    status = {"connected": None, "account": None}
    status_at = 0.0


async def save_state():
    try:
        await State.ctx.storage_state(path=str(STATE_PATH))
    except Exception:
        pass


async def make_context():
    kwargs = dict(viewport=VIEW, user_agent=UA)
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not API_KEYS:
        raise RuntimeError("Set the MY_API_KEY environment variable first.")
    State.pw = await async_playwright().start()
    State.browser = await State.pw.chromium.launch(headless=True, args=LAUNCH_ARGS)
    State.ctx = await make_context()
    State.ctx.on("page", _on_new_page)
    State.page = await State.ctx.new_page()
    try:
        await State.page.goto(SITE, wait_until="domcontentloaded", timeout=45000)
        await asyncio.sleep(1.5)
        await read_status(State.page)
    except Exception:
        pass
    tg_task = asyncio.create_task(telegram_loop())
    yield
    tg_task.cancel()
    await save_state()
    await State.browser.close()
    await State.pw.stop()


app = FastAPI(lifespan=lifespan)


def check_key(authorization):
    token = authorization[7:].strip() if authorization and authorization.lower().startswith("bearer ") else ""
    if not token or not any(secrets.compare_digest(token, k) for k in API_KEYS):
        raise HTTPException(401, "Invalid API key")


# ---------------- status ----------------
@app.get("/status")
async def status(authorization: str | None = Header(default=None)):
    check_key(authorization)
    if not State.lock.locked() and time.time() - State.status_at > 20:
        await refresh_status()
    return State.status


# ---------------- chat ----------------
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
            await asyncio.sleep(0.5)
    except Exception:
        pass
    return res


async def ask_deepseek(message: str, chat_url: str | None = None, think: bool = False,
                       search: bool = False, system: bool = True, extra_system: str = "") -> dict:
    page = State.page
    async with State.lock:  # one message at a time
        new_chat = not (chat_url and chat_url.startswith(SITE))
        target = SITE if new_chat else chat_url
        if new_chat or page.url != target:
            await page.goto(target, wait_until="domcontentloaded")
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
        prompt = with_system(message, extra_system) if (new_chat and system) else message

        g0 = await page.evaluate(JS_GRAB)
        before, before_think = g0["nAns"], g0["nThink"]
        box = page.locator(SEL_INPUT).first
        await box.click()
        await box.fill(prompt)
        await page.keyboard.press("Enter")

        # wait for the answer to start (thinking / searching can take a while)
        first_deadline = time.time() + (360 if think else 180 if search else 90)
        while True:
            g = await page.evaluate(JS_GRAB)
            if g["nAns"] > before and g["ans_text"].strip():
                break
            if time.time() > first_deadline:
                raise HTTPException(504, "No reply from DeepSeek (selectors changed, or it is busy).")
            await asyncio.sleep(0.6)

        # wait until the answer stops growing (finished streaming)
        need = 6 if (think or search) else 4
        last, stable = "", 0
        end = time.time() + 600
        while time.time() < end:
            g = await page.evaluate(JS_GRAB)
            cur = g["ans_text"]
            if cur and cur == last:
                stable += 1
                if stable >= need:
                    break
            else:
                stable, last = 0, cur
            await asyncio.sleep(0.5)

        reply = html_to_md(g["ans_html"]) or g["ans_text"].strip()
        thinking_html = g["think_html"] if (g["nThink"] > before_think) else ""
        State.status["connected"] = True
        await save_state()
        return {
            "reply": reply,
            "html": g["ans_html"],
            "thinking": html_to_md(thinking_html),
            "thinking_html": thinking_html,
            "chat_url": page.url,
            "features": features,
        }


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


class ChatIn(BaseModel):
    message: str
    chat_url: str | None = None
    think: bool = False
    search: bool = False
    system: bool = True


@app.post("/chat")
async def chat(body: ChatIn, authorization: str | None = Header(default=None)):
    check_key(authorization)
    return await respond_json(
        ask_deepseek(body.message, body.chat_url, body.think, body.search, body.system),
        lambda r: r)


# ---------------- OpenAI-compatible API ----------------
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
async def list_models(authorization: str | None = Header(default=None)):
    check_key(authorization)
    return {"object": "list", "data": [
        {"id": m, "object": "model", "created": 1700000000, "owned_by": "vortex"} for m in MODELS]}


def _oai_payload(model, res, prompt_len):
    msg = {"role": "assistant", "content": res["reply"]}
    if res.get("thinking"):
        msg["reasoning_content"] = res["thinking"]
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


async def _oai_stream(task, model):
    cid = f"chatcmpl-{secrets.token_hex(8)}"

    def chunk(delta, finish=None):
        return "data: " + json.dumps({
            "id": cid, "object": "chat.completion.chunk", "created": int(time.time()),
            "model": model, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}) + "\n\n"

    yield chunk({"role": "assistant", "content": ""})
    while True:
        d, _ = await asyncio.wait({task}, timeout=10)
        if d:
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
    if res.get("thinking"):
        yield chunk({"reasoning_content": res["thinking"]})
    text = res["reply"]
    for i in range(0, len(text), 160):
        yield chunk({"content": text[i:i + 160]})
    yield chunk({}, "stop")
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def oai(body: OAIIn, authorization: str | None = Header(default=None)):
    check_key(authorization)
    think, search = flags_from_model(body.model, body.think, body.search)
    prompt, extra = build_convo(body.messages)
    coro = ask_deepseek(prompt, None, think, search, True, extra)
    if body.stream:
        task = asyncio.create_task(coro)
        return StreamingResponse(_oai_stream(task, body.model), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
    return await respond_json(coro, lambda r: _oai_payload(body.model, r, len(prompt)))


# ---------------- Telegram bot (optional) ----------------
TG_STATE: dict = {}
TG_HELP = (
    f"{BOT_NAME} by {BOT_MAKER}: your coding partner.\n\n"
    "Just send a message, for example: build me a snake game in Python.\n\n"
    "/new - start a fresh chat\n"
    "/think on|off - deep thinking (slower, smarter)\n"
    "/search on|off - search the web\n"
    "/status - show settings\n"
    "/id - show your Telegram ID"
)


async def tg_send(call, chat_id, md):
    for chunk in split_md(md or "(empty reply)"):
        r = await call("sendMessage", chat_id=chat_id, text=md_to_tg(chunk), parse_mode="HTML",
                       disable_web_page_preview=True)
        if not r.get("ok"):   # formatting rejected -> send as plain text
            for i in range(0, len(chunk), 4000):
                await call("sendMessage", chat_id=chat_id, text=chunk[i:i + 4000])


async def handle_tg(call, allowed, upd):
    msg = upd.get("message") or {}
    text = (msg.get("text") or "").strip()
    chat_id = (msg.get("chat") or {}).get("id")
    uid = (msg.get("from") or {}).get("id")
    if not text or chat_id is None:
        return
    if text.split()[0].split("@")[0].lower() == "/id":
        await call("sendMessage", chat_id=chat_id, text=f"Your Telegram ID: {uid}")
        return
    if uid not in allowed:
        await call("sendMessage", chat_id=chat_id, text=(
            f"This is a private bot.\nYour Telegram ID is {uid}.\n"
            f"The owner must add it to TELEGRAM_ALLOWED_IDS on Render to let you in."))
        return
    st = TG_STATE.setdefault(chat_id, {"url": None, "think": False, "search": False})
    parts = text.split()
    cmd = parts[0].split("@")[0].lower() if text.startswith("/") else ""
    arg = parts[1].lower() if len(parts) > 1 else ""

    if cmd in ("/start", "/help"):
        await call("sendMessage", chat_id=chat_id, text=TG_HELP)
        return
    if cmd == "/new":
        st["url"] = None
        await call("sendMessage", chat_id=chat_id, text="New chat started.")
        return
    if cmd in ("/think", "/search"):
        key = cmd[1:]
        st[key] = (arg == "on") if arg in ("on", "off") else (not st[key])
        await call("sendMessage", chat_id=chat_id, text=f"{key.capitalize()}: {'ON' if st[key] else 'OFF'}")
        return
    if cmd == "/status":
        await call("sendMessage", chat_id=chat_id, text=(
            f"Think: {'ON' if st['think'] else 'OFF'}\nSearch: {'ON' if st['search'] else 'OFF'}\n"
            f"Chat: {'continuing' if st['url'] else 'new'}"))
        return
    if cmd:
        await call("sendMessage", chat_id=chat_id, text="Unknown command. Send /help")
        return

    async def typing():
        while True:
            await call("sendChatAction", chat_id=chat_id, action="typing")
            await asyncio.sleep(4)

    t = asyncio.create_task(typing())
    try:
        res = await ask_deepseek(text, st["url"], st["think"], st["search"])
        st["url"] = res["chat_url"]
        await tg_send(call, chat_id, res["reply"])
    except HTTPException as e:
        await call("sendMessage", chat_id=chat_id, text=f"Error: {e.detail}")
    except Exception as e:
        await call("sendMessage", chat_id=chat_id, text=f"Error: {e}")
    finally:
        t.cancel()


async def telegram_loop():
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        return
    allowed = {int(x) for x in re.findall(r"-?\d+", os.getenv("TELEGRAM_ALLOWED_IDS", ""))}
    base = f"https://api.telegram.org/bot{token}"
    offset = 0
    async with httpx.AsyncClient(timeout=45) as http:
        async def call(method, **kw):
            try:
                r = await http.post(f"{base}/{method}", json=kw)
                return r.json()
            except Exception:
                return {}

        await call("setMyCommands", commands=[
            {"command": "new", "description": "Start a fresh chat"},
            {"command": "think", "description": "Deep thinking on/off"},
            {"command": "search", "description": "Web search on/off"},
            {"command": "status", "description": "Show settings"},
            {"command": "help", "description": "Help"}])
        while True:
            data = await call("getUpdates", offset=offset, timeout=30, allowed_updates=["message"])
            if not data.get("ok"):
                await asyncio.sleep(4)
                continue
            for u in data.get("result", []):
                offset = u["update_id"] + 1
                asyncio.create_task(handle_tg(call, allowed, u))


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
@app.get("/")
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


@app.get("/import")
async def import_page():
    return FileResponse(HERE / "import.html")


@app.get("/health")
async def health():
    return {"ok": True}


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    host = "0.0.0.0" if os.getenv("PORT") else "127.0.0.1"
    uvicorn.run(app, host=host, port=port)
