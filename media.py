"""
Images for ShadowV1: AI image generation (proxied, provider hidden from users)
and picture search (Pinterest first, with safe fallbacks). SFW only.

All functions take an `http` object that behaves like httpx.AsyncClient.
"""
import hashlib
import hmac
import json
import os
import random
import re
import urllib.parse as up

BROWSER_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36")

STYLES = {
    "anime": "anime style, vibrant colors, detailed, high quality illustration",
    "realistic": "photorealistic, ultra detailed, natural lighting, 8k",
    "3d": "3d render, octane render, highly detailed",
    "cinematic": "cinematic still, dramatic lighting, shallow depth of field",
    "pixel": "pixel art, 16-bit, crisp pixels",
    "logo": "clean vector logo, minimal, flat design, white background",
    "painting": "oil painting, rich textures, masterpiece",
}


class MediaError(Exception):
    pass


# ---------- signed links (so <img src> works without sending the API key) ----------
def sign(secret: str, *parts) -> str:
    msg = "|".join(str(p) for p in parts).encode()
    return hmac.new(secret.encode(), msg, hashlib.sha256).hexdigest()[:24]


def img_path(secret: str, prompt: str, w=1024, h=1024, seed=0, style="") -> str:
    q = {"prompt": prompt, "w": w, "h": h, "seed": seed, "style": style,
         "sig": sign(secret, prompt, w, h, seed, style)}
    return "/media/img?" + up.urlencode(q)


# ---------- image generation ----------
def _attempts(prompt: str, w: int, h: int, seed: int, model: str | None):
    q = up.quote(prompt, safe="")
    key = os.getenv("POLLINATIONS_KEY", "").strip()
    out = []
    if key:
        wanted = [model or os.getenv("IMAGE_MODEL", "").strip() or "openai/gpt-image-2",
                  "black-forest-labs/flux.2-pro", "black-forest-labs/flux.1-schnell"]
        seen = set()
        for m in wanted:
            if m in seen:
                continue
            seen.add(m)
            out.append((f"https://gen.pollinations.ai/image/{q}",
                        {"model": m, "width": w, "height": h, "seed": seed, "safe": "true", "key": key}, 90))
    out.append((f"https://image.pollinations.ai/prompt/{q}",
                {"model": "flux", "width": w, "height": h, "seed": seed, "nologo": "true",
                 "enhance": "true", "safe": "true"}, 90))
    if not key:
        out.append((f"https://gen.pollinations.ai/image/{q}",
                    {"width": w, "height": h, "seed": seed, "safe": "true"}, 60))
    return out


async def generate_image(http, prompt: str, w: int = 1024, h: int = 1024, seed: int | None = None,
                         style: str = "", model: str | None = None) -> tuple[bytes, str]:
    """-> (image bytes, content-type). Tries the best model first, then falls back."""
    from shadow import is_unsafe_media_query
    if is_unsafe_media_query(prompt):
        raise MediaError("I only make safe-for-work images.")
    full = prompt.strip()
    if style and style.lower() in STYLES:
        full += ", " + STYLES[style.lower()]
    seed = random.randint(1, 2_000_000) if seed in (None, 0) else int(seed)
    w, h = max(256, min(int(w), 2048)), max(256, min(int(h), 2048))
    errors = []
    for url, params, timeout in _attempts(full[:1500], w, h, seed, model):
        try:
            r = await http.get(url, params=params, timeout=timeout, follow_redirects=True,
                               headers={"User-Agent": BROWSER_UA, "Accept": "image/*"})
            ctype = (r.headers.get("content-type") or "").split(";")[0].strip().lower()
            if r.status_code == 200 and ctype.startswith("image/") and len(r.content) > 2000:
                return r.content, ctype
            errors.append(f"{r.status_code}")
        except Exception as e:  # network / timeout
            errors.append(type(e).__name__)
    hint = "" if os.getenv("POLLINATIONS_KEY") else (
        " The free image service is rate-limited; add POLLINATIONS_KEY on Render for better and faster images.")
    raise MediaError("Image generation failed (" + ", ".join(errors) + ")." + hint)


# ---------- picture search ----------
def _upsize(u: str) -> str:
    return re.sub(r"i\.pinimg\.com/(?:\d+x(?:\d+)?|originals)/", "i.pinimg.com/736x/", u)


def _pin_from_result(r: dict):
    imgs = r.get("images") or {}
    best = None
    for k in ("orig", "736x", "564x", "474x", "236x"):
        if isinstance(imgs.get(k), dict) and imgs[k].get("url"):
            best = imgs[k]["url"]
            break
    if not best:
        return None
    pid = r.get("id") or ""
    return {"url": best, "title": (r.get("grid_title") or r.get("title") or "").strip(),
            "link": f"https://www.pinterest.com/pin/{pid}/" if pid else (r.get("link") or ""),
            "source": "pinterest"}


async def _pinterest_json(http, query: str, n: int):
    data = {"options": {"query": query, "scope": "pins", "bookmarks": [""], "page_size": max(n * 3, 25)},
            "context": {}}
    params = {"source_url": f"/search/pins/?q={up.quote(query)}&rs=typed", "data": json.dumps(data)}
    r = await http.get("https://www.pinterest.com/resource/BaseSearchResource/get/", params=params, timeout=20,
                       headers={"User-Agent": BROWSER_UA, "Accept": "application/json, text/javascript, */*",
                                "X-Requested-With": "XMLHttpRequest", "X-Pinterest-AppState": "active",
                                "Accept-Language": "en-US,en;q=0.9"})
    if r.status_code != 200:
        return []
    results = (((r.json().get("resource_response") or {}).get("data") or {}).get("results")) or []
    return [p for p in (_pin_from_result(x) for x in results if isinstance(x, dict)) if p]


async def _pinterest_html(http, query: str):
    r = await http.get("https://www.pinterest.com/search/pins/", params={"q": query, "rs": "typed"}, timeout=20,
                       headers={"User-Agent": BROWSER_UA, "Accept-Language": "en-US,en;q=0.9"},
                       follow_redirects=True)
    if r.status_code != 200:
        return []
    urls = re.findall(r"https://i\.pinimg\.com/(?:\d+x(?:\d+)?|originals)/[\w/\-]+\.(?:jpg|jpeg|png|webp)", r.text)
    return [{"url": _upsize(u), "title": "", "link": "", "source": "pinterest"} for u in urls]


async def _wallhaven(http, query: str, n: int):
    anime = bool(re.search(r"\b(anime|manga|naruto|one piece|goku|dragon ball|jujutsu|demon slayer|"
                           r"waifu|ghibli|attack on titan|bleach|chainsaw|spy x family)\b", query, re.I))
    r = await http.get("https://wallhaven.cc/api/v1/search", timeout=20,
                       params={"q": query, "categories": "010" if anime else "111", "purity": "100",
                               "sorting": "relevance", "order": "desc"},
                       headers={"User-Agent": BROWSER_UA})
    if r.status_code != 200:
        return []
    out = []
    for x in (r.json().get("data") or []):
        if x.get("path"):
            out.append({"url": x["path"], "title": "", "link": x.get("url", ""), "source": "wallhaven"})
    return out


async def search_pictures(http, query: str, n: int = 5) -> list[dict]:
    """Best-effort picture search. Pinterest first, then fallbacks. Always SFW."""
    from shadow import is_unsafe_media_query
    if is_unsafe_media_query(query):
        raise MediaError("I only fetch safe-for-work pictures.")
    n = max(1, min(int(n), 20))
    seen, out = set(), []
    for fn in (_pinterest_json, _pinterest_html, _wallhaven):
        try:
            items = await (fn(http, query, n) if fn is not _pinterest_html else fn(http, query))
        except Exception:
            items = []
        for it in items:
            u = it["url"]
            if u not in seen:
                seen.add(u)
                out.append(it)
        if len(out) >= n:
            break
    if not out:
        raise MediaError("Couldn't find pictures for that right now. Try different words.")
    return out[:n]


async def fetch_bytes(http, url: str) -> tuple[bytes, str]:
    r = await http.get(url, timeout=30, follow_redirects=True,
                       headers={"User-Agent": BROWSER_UA, "Referer": "https://www.pinterest.com/"})
    ctype = (r.headers.get("content-type") or "").split(";")[0].strip().lower()
    if r.status_code != 200 or not ctype.startswith("image/"):
        raise MediaError(f"download failed ({r.status_code})")
    return r.content, ctype
