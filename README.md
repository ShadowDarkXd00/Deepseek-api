# ShadowV1 provider

Your DeepSeek web-chat account as an OpenAI-compatible API, hosted on Render.
Use it as a custom provider in VS Code (Cline, Roo, Continue), OpenCode, AntiGem and any app.

- `POST /v1/chat/completions`: chat, streaming, and native tool calling (`tools` -> `tool_calls`) for coding agents
- `GET /v1/models`: `shadowv1`, `shadowv1-think`, `shadowv1-search`, `shadowv1-think-search`
- `/`: info page (base URL, key, copy buttons, test) | `/connect`, `/remote`: sign in to DeepSeek | `/health`, `/status`, `/diag`

Env vars: `MY_API_KEY` (required; comma separates several keys), `REDIS_URL` (Render Key Value, keeps the DeepSeek login).

The full bot (Telegram, AI agent, GitHub/Render tools, images, pictures) is kept on the branch `full-bot-backup`.
