"""
GitHub + Render integration for ShadowV1: push generated files, run workflows,
deploy, read logs, and host a new repo as a web service.

Tokens come from env vars (set them yourself in the Render dashboard):
  GITHUB_TOKEN     GitHub personal access token (repo + workflow scopes)
  RENDER_API_KEY   Render API key (Account Settings > API Keys)
`http` behaves like httpx.AsyncClient.
"""
import base64
import os

GH = "https://api.github.com"
RENDER = "https://api.render.com/v1"


class IntegrationError(Exception):
    pass


def _msg(r) -> str:
    try:
        j = r.json()
        if isinstance(j, dict):
            m = j.get("message") or j.get("error") or ""
            errs = j.get("errors")
            if errs:
                m += " " + str(errs)[:200]
            return str(m) or r.text[:200]
    except Exception:
        pass
    return (getattr(r, "text", "") or "")[:200]


class GitHub:
    def __init__(self, http, token: str | None = None):
        self.http = http
        self.token = token or os.getenv("GITHUB_TOKEN", "").strip()

    def _need(self):
        if not self.token:
            raise IntegrationError("GitHub isn't connected. Add GITHUB_TOKEN on Render "
                                   "(GitHub > Settings > Developer settings > Personal access tokens).")

    async def _req(self, method, path, ok=(200, 201, 202, 204), **kw):
        self._need()
        r = await self.http.request(
            method, GH + path, timeout=30,
            headers={"Authorization": f"Bearer {self.token}", "Accept": "application/vnd.github+json",
                     "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "ShadowV1"}, **kw)
        if r.status_code not in ok:
            raise IntegrationError(f"GitHub {r.status_code}: {_msg(r)}")
        return r

    async def me(self) -> str:
        return (await self._req("GET", "/user")).json().get("login", "")

    async def repos(self, limit=20) -> list[dict]:
        r = await self._req("GET", "/user/repos", params={"per_page": limit, "sort": "pushed"})
        return [{"name": x["full_name"], "private": x.get("private", False), "url": x["html_url"]}
                for x in r.json()]

    async def create_repo(self, name: str, private: bool = True, description: str = "") -> dict:
        r = await self._req("POST", "/user/repos",
                            json={"name": name, "private": private, "description": description[:200],
                                  "auto_init": False})
        j = r.json()
        return {"name": j["full_name"], "url": j["html_url"], "default_branch": j.get("default_branch", "main")}

    async def commit_files(self, repo: str, files: list[tuple[str, str]], message: str,
                           branch: str | None = None) -> dict:
        if not files:
            raise IntegrationError("No files to push (the answer had no code blocks with file paths).")
        info = (await self._req("GET", f"/repos/{repo}")).json()
        branch = branch or info.get("default_branch") or "main"
        ref = await self._req("GET", f"/repos/{repo}/git/ref/heads/{branch}", ok=(200, 404, 409))
        if ref.status_code != 200:          # empty repo: first commit through the Contents API
            sha = ""
            for path, content in files:
                body = {"message": message, "content": base64.b64encode(content.encode()).decode(),
                        "branch": branch}
                r = await self._req("PUT", f"/repos/{repo}/contents/{path}", json=body)
                sha = ((r.json().get("commit") or {}).get("sha")) or sha
            return {"repo": repo, "branch": branch, "files": len(files), "commit": sha,
                    "url": f"https://github.com/{repo}"}
        head = ref.json()["object"]["sha"]
        base_tree = (await self._req("GET", f"/repos/{repo}/git/commits/{head}")).json()["tree"]["sha"]
        items = []
        for path, content in files:
            b = (await self._req("POST", f"/repos/{repo}/git/blobs",
                                 json={"content": base64.b64encode(content.encode()).decode(),
                                       "encoding": "base64"})).json()
            items.append({"path": path, "mode": "100755" if path.endswith(".sh") else "100644",
                          "type": "blob", "sha": b["sha"]})
        tree = (await self._req("POST", f"/repos/{repo}/git/trees",
                                json={"base_tree": base_tree, "tree": items})).json()
        commit = (await self._req("POST", f"/repos/{repo}/git/commits",
                                  json={"message": message, "tree": tree["sha"], "parents": [head]})).json()
        await self._req("PATCH", f"/repos/{repo}/git/refs/heads/{branch}", json={"sha": commit["sha"]})
        return {"repo": repo, "branch": branch, "files": len(files), "commit": commit["sha"],
                "url": f"https://github.com/{repo}/commit/{commit['sha']}"}

    async def runs(self, repo: str, n=5) -> list[dict]:
        r = await self._req("GET", f"/repos/{repo}/actions/runs", params={"per_page": n})
        return [{"name": x.get("name"), "status": x.get("status"), "conclusion": x.get("conclusion"),
                 "branch": x.get("head_branch"), "url": x.get("html_url")}
                for x in r.json().get("workflow_runs", [])]

    async def dispatch(self, repo: str, workflow: str, ref: str | None = None, inputs: dict | None = None):
        if not ref:
            ref = (await self._req("GET", f"/repos/{repo}")).json().get("default_branch", "main")
        body = {"ref": ref}
        if inputs:
            body["inputs"] = inputs
        await self._req("POST", f"/repos/{repo}/actions/workflows/{workflow}/dispatches", json=body)
        return {"repo": repo, "workflow": workflow, "ref": ref}


class RenderAPI:
    def __init__(self, http, key: str | None = None):
        self.http = http
        self.key = key or os.getenv("RENDER_API_KEY", "").strip()

    def _need(self):
        if not self.key:
            raise IntegrationError("Render isn't connected. Add RENDER_API_KEY on Render "
                                   "(Account Settings > API Keys).")

    async def _req(self, method, path, ok=(200, 201, 202, 204), **kw):
        self._need()
        r = await self.http.request(method, RENDER + path, timeout=30,
                                    headers={"Authorization": f"Bearer {self.key}", "Accept": "application/json"},
                                    **kw)
        if r.status_code not in ok:
            raise IntegrationError(f"Render {r.status_code}: {_msg(r)}")
        return r

    async def services(self, limit=50) -> list[dict]:
        r = await self._req("GET", "/services", params={"limit": limit})
        out = []
        for item in r.json():
            s = item.get("service", item)
            out.append({"id": s.get("id"), "name": s.get("name"), "type": s.get("type"),
                        "owner": s.get("ownerId"), "repo": s.get("repo", ""),
                        "url": (s.get("serviceDetails") or {}).get("url", ""),
                        "dashboard": s.get("dashboardUrl", "")})
        return out

    async def find(self, name_or_id: str) -> dict:
        q = (name_or_id or "").strip().lower()
        svcs = await self.services()
        for s in svcs:
            if q in (s["id"].lower(), (s["name"] or "").lower()):
                return s
        for s in svcs:
            if q and q in (s["name"] or "").lower():
                return s
        raise IntegrationError(f"No Render service matching '{name_or_id}'.")

    async def deploy(self, service_id: str, clear_cache: bool = False) -> dict:
        r = await self._req("POST", f"/services/{service_id}/deploys",
                            json={"clearCache": "clear" if clear_cache else "do_not_clear"})
        j = r.json() if r.content else {}
        return {"deploy": j.get("id", ""), "status": j.get("status", "queued")}

    async def deploys(self, service_id: str, n=3) -> list[dict]:
        r = await self._req("GET", f"/services/{service_id}/deploys", params={"limit": n})
        out = []
        for item in r.json():
            d = item.get("deploy", item)
            out.append({"id": d.get("id"), "status": d.get("status"), "created": d.get("createdAt"),
                        "message": ((d.get("commit") or {}).get("message") or "")[:80]})
        return out

    async def logs(self, service_id: str, owner_id: str, n=40) -> list[str]:
        r = await self._req("GET", "/logs", params={"ownerId": owner_id, "resource": service_id,
                                                    "limit": n, "direction": "backward"})
        return [x.get("message", "") for x in reversed(r.json().get("logs", []))]

    async def create_web_service(self, name: str, repo_url: str, owner_id: str, runtime: str = "docker",
                                 branch: str = "main", plan: str = "free", region: str = "singapore",
                                 env: dict | None = None, build: str = "", start: str = "") -> dict:
        details = {"runtime": runtime, "plan": plan, "region": region}
        if runtime == "docker":
            details["envSpecificDetails"] = {"dockerfilePath": "./Dockerfile"}
        elif runtime == "node":
            details["envSpecificDetails"] = {"buildCommand": build or "npm install",
                                             "startCommand": start or "npm start"}
        else:
            details["envSpecificDetails"] = {"buildCommand": build or "pip install -r requirements.txt",
                                             "startCommand": start or "python app.py"}
        body = {"type": "web_service", "name": name, "ownerId": owner_id, "repo": repo_url, "branch": branch,
                "autoDeploy": "yes", "serviceDetails": details}
        if env:
            body["envVars"] = [{"key": k, "value": v} for k, v in env.items()]
        r = await self._req("POST", "/services", json=body)
        j = r.json()
        s = j.get("service", j)
        return {"id": s.get("id"), "name": s.get("name"),
                "url": (s.get("serviceDetails") or {}).get("url", ""), "dashboard": s.get("dashboardUrl", "")}
