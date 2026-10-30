# Publishing the dashboard to students

Two decisions come before any deployment:

1. **Where it runs.** The dashboard executes learner Python *in the server
   process*. That is fine on a host you control; it is not fine on a shared
   machine holding other data.
2. **Who can reach it.** A public URL plus a cohort access code is enough for a
   class. It is not authentication, and it does not make the app safe to expose
   to the open internet indefinitely.

Everything below assumes the project folder is the deployment unit.

---

## Option A — Streamlit Community Cloud (free, fastest)

Best for a single cohort and a term's teaching. The free tier gives roughly 1 GB
of RAM per app, which is ample: each task runs in milliseconds.

1. Push the project to a GitHub repository (public or private both work).
   `app.py` and `requirements.txt` must be in the same folder.
2. Sign in at <https://share.streamlit.io> with GitHub and authorise the repo.
3. **Create app** → pick repository and branch → set *Main file path* to `app.py`.
4. **Advanced settings** → Python 3.12 → paste the access code into *Secrets*:

   ```toml
   APP_PASSWORD = "cohort-2026-code"
   ```

5. **Deploy.** The first build takes two to five minutes.
6. Share the resulting `https://<name>.streamlit.app` link.

Operational notes:

- Idle apps sleep. A student arriving after a quiet night waits ~30 s for the
  container to wake, then a few more seconds for the statsmodels import. Warn
  them, or keep the app warm with a scheduled ping.
- Reboot and logs are in the app's dashboard; secrets can be edited without a
  redeploy.
- The URL is public even for a private repo, which is exactly why `APP_PASSWORD`
  exists.

---

## Option B — Your own server, in Docker (control and isolation)

Use this when the app must sit on university infrastructure, or when you want a
hard memory and CPU ceiling per instance.

```bash
docker build -t analytics-lab .
docker run -d --name analytics-lab \
  -p 8501:8501 \
  --memory=1g --cpus=1 \
  --read-only --tmpfs /tmp \
  -e APP_PASSWORD='cohort-2026-code' \
  analytics-lab
```

Then terminate TLS in front of it. Caddy is the least effort and proxies
WebSockets automatically:

```caddyfile
lab.example.ac.uk {
    reverse_proxy localhost:8501
}
```

With nginx, the WebSocket upgrade headers are mandatory or the page will load
but never connect:

```nginx
location / {
    proxy_pass http://localhost:8501;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
    proxy_set_header Host $host;
    proxy_read_timeout 86400;
}
```

`--read-only` with a `tmpfs` for `/tmp` is worth keeping: it stops a submission
that writes a file from persisting anything.

---

## Option C — University platform or LMS

- If your institution runs Streamlit, Shiny or JupyterHub as a service, deploy
  there. You inherit single sign-on, network isolation and backups.
- **Do not embed the app in an LMS iframe by default.** Streamlit sends frame
  headers that block it, and weakening them is a security change, not a
  configuration tweak. A plain link from the module page is more robust and
  easier to support.

---

## Option D — Temporary sharing (demo only)

For a live session, a tunnel avoids deploying anything:

```bash
cloudflared tunnel --url http://localhost:8501
```

The URL dies with your laptop. Use it for a walkthrough, not for assessment.

---

## Operating notes

| Concern | Behaviour |
| --- | --- |
| Student isolation | Each browser session has its own code, output and progress. Nobody sees anyone else's answers. |
| Persistence | Progress lives in the session. A restart clears it; students export answers from the **Progress** page. |
| Concurrency | Each submission runs in a thread capped at 6 seconds. Budget roughly 150–250 MB per active session. |
| Cold start | ~30 s on slow storage because statsmodels is imported at start-up. |
| Production flags | Keep `--server.fileWatcherType none`; the watcher is a development feature. |
| No data collected | The app stores nothing about learners beyond the session and sends no analytics. |

---

## Security position — read before publishing

The runner restricts builtins and imports, and caps execution time. Those
measures stop accidents: `import os`, `open()`, a runaway loop. **They are not a
security boundary.** A determined student can escape a pure-Python `exec`, and
that would give them the server process.

Consequences:

- Publish on a managed platform that isolates each app, or on a host you own and
  can rebuild.
- Keep the container limits (memory, CPU, read-only filesystem) even if the host
  looks private.
- Treat `APP_PASSWORD` as a door code, not authentication: rotate it each
  cohort, and never reuse a password you use elsewhere.
- If the cohort is genuinely untrusted (open enrolment, external assessment),
  replace `exec` with one container per submission, or delegate execution to a
  sandbox service such as Piston or Judge0. The marking layer in `engine.py`
  would not need to change — only `run_code`.
