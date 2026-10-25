# ShadowV1 by Vortex (unofficial DeepSeek web-chat API)

Your own website, OpenAI-compatible API and Telegram bot on top of the DeepSeek web chat.

## Deploy on Render
- New > Web Service > connect this repo > Runtime: Docker
- Env var `MY_API_KEY` = a long random password (comma-separate several keys to give each bot its own)
- Recommended: Starter plan + a Disk mounted at `/data` (keeps the DeepSeek login after restarts)

## Connect your DeepSeek account
Open `/connect` (sign in with your own Chrome) or `/remote` (live browser view).

## Use it
- Website: `/` (new chats, old chats, code blocks with Copy, Think and Search switches)
- API guide with your URL and key filled in: `/guide`
- OpenAI-compatible: `POST /v1/chat/completions`, `GET /v1/models`
  - models: `shadowv1`, `shadowv1-think`, `shadowv1-search`, `shadowv1-think-search`
  - or send `"think": true` / `"search": true`
  - `"stream": true` is supported
- Simple: `POST /chat` `{"message": "...", "chat_url": null, "think": false, "search": false}`

## Telegram bot
Env vars: `TELEGRAM_BOT_TOKEN` (from @BotFather) and `TELEGRAM_ALLOWED_IDS` (comma-separated Telegram IDs).
Message the bot once to see your ID. Commands: `/new`, `/think on|off`, `/search on|off`, `/status`.

## Optional env vars
`BOT_NAME` (default ShadowV1), `BOT_MAKER` (default Vortex), `SYSTEM_PROMPT` (replace the built-in coding prompt).

## If it breaks
DeepSeek changes its page layout. Update `SEL_INPUT` / `SEL_REPLY` in `server.py`, and the labels in `shadow.py`.
