"""
Unofficial DeepSeek web-chat -> your own website + API key.

  /          chat website (new chats, old chats, code blocks)
  /remote    sign in to DeepSeek (Google) through a live browser view
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
VIEW = {"width": 1000, "height": 700}
LAUNCH_ARGS = [
    "--disable-blink-features=AutomationControlled",
    "--no-sandbox",
    "--disable-dev-shm-usage",
]
ALLOWED_KEYS = {"Enter", "Tab", "Backspace", "Escape", "ArrowDown", "ArrowUp",
                "ArrowLeft", "ArrowRight", "Delete", "Space"}
# ----------------------------


class State:
    pw = None
    browser = None
    ctx = None
    page = None
    login_pages: list = []
    login_active = False
    lock = asyncio.Lock()


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


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not API_KEY:
        raise RuntimeError("Set the MY_API_KEY environment variable first.")
    State.pw = await async_playwright().start()
    State.browser = await State.pw.chromium.launch(headless=True, args=LAUNCH_ARGS)
    State.ctx = await make_context()
    # popups (e.g. the Google sign-in window) become part of the remote view
    State.ctx.on("page", lambda p: State.login_pages.append(p) if State.login_active else None)
    State.page = await State.ctx.new_page()
    yield
    await save_state()
    await State.browser.close()
    await State.pw.stop()


app = FastAPI(lifespan=lifespan)


def check_key(authorization):
    expected = f"Bearer {API_KEY}"
    if not authorization or not secrets.compare_digest(authorization, expected):
        raise HTTPException(401, "Invalid API key")


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
            raise HTTPException(409, "Not logged in to DeepSeek. Open /remote and sign in.")
        if "sign_in" in page.url:
            raise HTTPException(409, "Not logged in to DeepSeek. Open /remote and sign in.")

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


# ---------------- remote login view ----------------
def active_login_page():
    pages = [p for p in State.login_pages if not p.is_closed()]
    return pages[-1] if pages else None


@app.post("/remote/open")
async def remote_open(authorization: str | None = Header(default=None)):
    check_key(authorization)
    State.login_active = True
    if active_login_page() is None:
        p = await State.ctx.new_page()
        State.login_pages = [p]
    await State.login_pages[0].goto(SITE, wait_until="domcontentloaded")
    return {"ok": True}


@app.get("/remote/shot")
async def remote_shot(authorization: str | None = Header(default=None)):
    check_key(authorization)
    p = active_login_page()
    if p is None:
        raise HTTPException(409, "Press Start first.")
    img = await p.screenshot(type="jpeg", quality=55)
    return Response(img, media_type="image/jpeg")


class ClickIn(BaseModel):
    x: float
    y: float


class TypeIn(BaseModel):
    text: str


class KeyIn(BaseModel):
    key: str


@app.post("/remote/click")
async def remote_click(body: ClickIn, authorization: str | None = Header(default=None)):
    check_key(authorization)
    p = active_login_page()
    if p is None:
        raise HTTPException(409, "Press Start first.")
    await p.mouse.click(body.x, body.y)
    return {"ok": True}


@app.post("/remote/type")
async def remote_type(body: TypeIn, authorization: str | None = Header(default=None)):
    check_key(authorization)
    p = active_login_page()
    if p is None:
        raise HTTPException(409, "Press Start first.")
    await p.keyboard.type(body.text, delay=25)
    return {"ok": True}


@app.post("/remote/key")
async def remote_key(body: KeyIn, authorization: str | None = Header(default=None)):
    check_key(authorization)
    if body.key not in ALLOWED_KEYS:
        raise HTTPException(400, "Key not allowed")
    p = active_login_page()
    if p is None:
        raise HTTPException(409, "Press Start first.")
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
    # reload the chat page so it picks up the new login
    try:
        await State.page.goto(SITE, wait_until="domcontentloaded")
    except Exception:
        pass
    return {"ok": True}


# ---------------- pages ----------------
@app.get("/")
async def index():
    return FileResponse(HERE / "index.html")


@app.get("/remote")
async def remote_page():
    return FileResponse(HERE / "remote.html")


@app.get("/health")
async def health():
    return {"ok": True}


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    host = "0.0.0.0" if os.getenv("PORT") else "127.0.0.1"
    uvicorn.run(app, host=host, port=port)
