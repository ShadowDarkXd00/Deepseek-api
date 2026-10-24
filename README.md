# My DeepSeek Code Chat (unofficial)

Your own website + API key on top of the DeepSeek web chat.

## Files to upload to a PRIVATE GitHub repo
server.py, index.html, remote.html, Dockerfile, requirements.txt, .gitignore

## Deploy on Render
- New > Web Service > connect the repo > Runtime: Docker
- Environment variable: MY_API_KEY = a long random password (this is your API key)
- Recommended: Starter plan + a Disk mounted at /data (keeps your login after restarts)

## First use
1. Open https://YOUR-APP.onrender.com/remote
2. Enter your API key, press Start, tap through the Google sign-in, press "I'm signed in"
3. Go to the main page and chat

## API
POST /chat  {"message": "write a python hello world", "chat_url": null}
Header: Authorization: Bearer YOUR_KEY
Send the returned chat_url back to continue the same conversation.

OpenAI-style: POST /v1/chat/completions

## If it breaks
DeepSeek changes its page layout. Update SEL_INPUT / SEL_REPLY in server.py.
