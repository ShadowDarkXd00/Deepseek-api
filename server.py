"""
Unofficial DeepSeek web-chat -> your own website + API key.

  /          chat website (new chats, old chats, code blocks)
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
import secrets
import time
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Header, HTTPException, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel
from playwright.async_api import async_playwright

# ---------- CONFIG ----------
API_KEY = os.getenv("MY_API_KEY", "")
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
    if not API_KEY:
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
    yield
    await save_state()
    await State.browser.close()
    await State.pw.stop()


app = FastAPI(lifespan=lifespan)


def check_key(authorization):
    expected = f"Bearer {API_KEY}"
    if not authorization or not secrets.compare_digest(authorization, expected):
        raise HTTPException(401, "Invalid API key")


# ---------------- status ----------------
@app.get("/status")
async def status(authorization: str | None = Header(default=None)):
    check_key(authorization)
    if not State.lock.locked() and time.time() - State.status_at > 20:
        await refresh_status()
    return State.status


# ---------------- chat ----------------
async def ask_deepseek(message: str, chat_url: str | None = None) -> dict:
    page = State.page
    async with State.lock:  # one message at a time
        target = chat_url if chat_url and chat_url.startswith(SITE) else SITE
        if page.url != target:
            await page.goto(target, wait_until="domcontentloaded")
        try:
            await page.wait_for_selector(SEL_INPUT, timeout=25000)
        except Exception:
            State.status = {"connected": False, "account": None}
            raise HTTPException(409, "Not logged in to DeepSeek. Connect your account first.")
        if "sign_in" in page.url:
            State.status = {"connected": False, "account": None}
            raise HTTPException(409, "Not logged in to DeepSeek. Connect your account first.")

        before = await page.locator(SEL_REPLY).count()
        box = page.locator(SEL_INPUT).first
        await box.click()
        await box.fill(message)
        await page.keyboard.press("Enter")

        deadline = time.time() + 60
        while await page.locator(SEL_REPLY).count() <= before:
            if time.time() > deadline:
                raise HTTPException(504, "No reply from DeepSeek (selectors changed?)")
            await asyncio.sleep(0.5)

        last, stable = "", 0
        deadline = time.time() + 240
        while time.time() < deadline:
            text = await page.locator(SEL_REPLY).last.inner_text()
            if text and text == last:
                stable += 1
                if stable >= 4:  # ~2s unchanged = finished streaming
                    break
            else:
                stable, last = 0, text
            await asyncio.sleep(0.5)

        html = await page.locator(SEL_REPLY).last.inner_html()
        url = page.url
        State.status["connected"] = True
        await save_state()
        return {"reply": last, "html": html, "chat_url": url}


class ChatIn(BaseModel):
    message: str
    chat_url: str | None = None


@app.post("/chat")
async def chat(body: ChatIn, authorization: str | None = Header(default=None)):
    check_key(authorization)
    return await ask_deepseek(body.message, body.chat_url)


class OAIMessage(BaseModel):
    role: str
    content: str


class OAIIn(BaseModel):
    model: str = "deepseek-web"
    messages: list[OAIMessage]


@app.post("/v1/chat/completions")
async def oai(body: OAIIn, authorization: str | None = Header(default=None)):
    check_key(authorization)
    prompt = next((m.content for m in reversed(body.messages) if m.role == "user"), "")
    res = await ask_deepseek(prompt)
    return {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body.model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": res["reply"]},
                     "finish_reason": "stop"}],
    }


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
    code: str
    data: str


@app.post("/session/import")
async def session_import(body: ImportIn):
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
