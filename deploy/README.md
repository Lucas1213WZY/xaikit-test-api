# Deploying the study server on a university / lab server

Three things decide the setup, and you can answer all three in one paste on the
server:

```bash
# 1. Is Docker available to you?
docker info >/dev/null 2>&1 && echo "docker: yes" || echo "docker: no"

# 2. Can you run services and keep them alive after logout?
systemctl --user status >/dev/null 2>&1 && echo "user systemd: yes" || echo "user systemd: no"
sudo -n true 2>/dev/null && echo "sudo: yes" || echo "sudo: no"

# 3. Is anything reachable from outside?
hostname -f
curl -s ifconfig.me; echo          # empty or a private 10./192.168. address
                                   # means you are behind the campus network
```

Then follow the matching path below. **Before any of them**, stage a copy with
the human participant data removed — see "Human data" at the bottom.

---

## Path A — Docker is available

Everything is already written. Copy the staged tree plus `Dockerfile`,
`docker-compose.yml` and `Caddyfile` to the server, fill in `.env`, and:

```bash
docker compose up -d --build
```

If the institution already terminates TLS for you (most do — a central reverse
proxy fronting departmental services), drop the `caddy` service from
`docker-compose.yml` and publish the API port on localhost only:

```yaml
  api:
    ports:
      - "127.0.0.1:8000:8000"
```

then give your admin the proxy target `http://<host>:8000`.

## Path B — No Docker (conda + systemd)

The common case on shared research machines.

```bash
# on the server, from the staged tree at ~/xaikit-api
conda create -n xaikit-api python=3.10 -y
conda activate xaikit-api
pip install -r requirements.txt
pip install "fastapi>=0.115,<1" "uvicorn>=0.30,<1"

cat > ~/xaikit-api/.env <<'EOF'
XAIKIT_API_TOKEN=<python -c "import secrets; print(secrets.token_urlsafe(32))">
XAIKIT_ALLOWED_ORIGINS=https://you.github.io
XAIKIT_SERVER_RUNS_DIR=/home/<you>/xaikit-api/server_runs
EOF
chmod 600 ~/xaikit-api/.env

# quick check before making it a service
uvicorn server.app:app --host 127.0.0.1 --port 8000
curl -s localhost:8000/api/health
```

Then install [`xaikit-api.service`](xaikit-api.service) as a user service — the
header of that file has the exact commands. Ask your admin for
`loginctl enable-linger $USER` so it survives logout.

No systemd at all? `tmux new -s xaikit` then run uvicorn inside it. It survives
logout but not a reboot, so treat it as temporary.

## Path C — Behind the campus firewall

If step 3 above showed a private address, or inbound 443 is closed, a UI on
GitHub Pages **cannot reach the server** no matter how it is configured. A
browser on the open internet has no route in. Options, in order of preference:

1. **Ask for a hostname and inbound HTTPS.** Departments usually have a process
   for this. You want: a DNS name, inbound 443, and either a certificate or a
   central proxy. Say it is an internal research API.
2. **Keep everything inside the network.** Host the UI on the same server
   (Caddy can serve the static files next to the API) and require VPN. Nothing
   leaves the network — often the right answer for a study touching human data.
3. **A tunnel**, if policy allows it: `cloudflared tunnel` on the lab server
   gives a public HTTPS URL without inbound ports. Check with your admin
   first — it deliberately bypasses the firewall, and doing that unilaterally
   on institutional hardware is the kind of thing that gets accounts suspended.

### Mounted under a path prefix

If the central proxy serves you at `https://cs.example.edu/xaikit/` rather than
on its own hostname, tell uvicorn, or every URL the API generates will be wrong:

```bash
uvicorn server.app:app --host 127.0.0.1 --port 8000 --root-path /xaikit
```

---

## Human data

`.dockerignore` only filters Docker builds. A `git clone` or plain `rsync` of
this repo onto the server carries the participant data with it —
`assets/human_data/` and the 66 MB of per-participant records under
`src/cognitive_models/CoAX/results/` — and on a shared machine other accounts
may be able to read your home directory.

Deploy from a staged tree instead:

```bash
./deploy/stage_deploy.sh /tmp/xaikit-deploy
rsync -av --delete /tmp/xaikit-deploy/ user@server:~/xaikit-api/
```

The script removes every human-data path and refuses to finish if any survive.
The full pipeline was verified end to end against exactly this tree, so nothing
the server does needs those files.

On the server, keep the tree to yourself:

```bash
chmod 700 ~/xaikit-api
chmod 600 ~/xaikit-api/.env
```

Results the server generates are simulated CoAX responses, not human data — but
if you later run real participants through this UI, the artifacts under
`server_runs/` become human data, and where that server lives stops being only
a convenience question. Check what your ethics approval says about where
participant data may be stored before that point, not after.
