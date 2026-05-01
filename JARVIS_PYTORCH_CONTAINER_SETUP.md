# Jarvis PyTorch Container Setup

These are the exact steps validated on a fresh Jarvis Labs PyTorch container on
May 1, 2026.

Validated environment:

- Jarvis instance: fresh `pytorch` container
- Region: `IN2`
- GPU: `L4`
- Port: `6006`, exposed by the PyTorch template by default
- Backend branch: `svishnu88/ComfyUI`, `jarvis-main`
- Frontend branch: `svishnu88/ComfyUI_frontend`, `jarvis-main`
- Python environment: `uv venv --system-site-packages`
- Node: manually installed to `/home/node`

Do not pass `--http-ports 6006` to `jl create`; the CLI rejects it because
`6006` is already reserved/exposed for this template.

## 1. Create A Fresh PyTorch Container

Run this locally:

```bash
jl create \
  --gpu L4 \
  --template pytorch \
  --region IN2 \
  --storage 80 \
  --name comfyui-jarvis-manual-test \
  --yes \
  --json
```

Save the returned `machine_id`. The public ComfyUI URL is the first item in
`endpoints`.

## 2. Install ComfyUI And Build The Jarvis Frontend

Run this on the machine with `jl exec <machine_id> -- bash -lc '...'`, or SSH
into the machine and run the body directly.

```bash
set -euxo pipefail

cd /home

rm -rf \
  /home/ComfyUI \
  /home/ComfyUI_frontend \
  /home/comfyui-venv \
  /home/node \
  /home/node.tar.xz

git clone -b jarvis-main https://github.com/svishnu88/ComfyUI.git /home/ComfyUI
git clone -b jarvis-main https://github.com/svishnu88/ComfyUI_frontend.git /home/ComfyUI_frontend

uv venv --system-site-packages /home/comfyui-venv
source /home/comfyui-venv/bin/activate
uv pip install -r /home/ComfyUI/requirements.txt

curl -fsSL https://nodejs.org/dist/v24.15.0/node-v24.15.0-linux-x64.tar.xz -o /home/node.tar.xz
tar -xf /home/node.tar.xz
mv /home/node-v24.15.0-linux-x64 /home/node

export PATH=/home/node/bin:$PATH
node --version
corepack enable
pnpm --version

cd /home/ComfyUI_frontend
pnpm install --frozen-lockfile
DISTRIBUTION=jarvis GENERATE_SOURCEMAP=false pnpm exec vite build --config vite.config.mts

cd /home/ComfyUI
python -m py_compile server.py comfy_api/feature_flags.py
```

## 3. Start ComfyUI

For a manual SSH session:

```bash
cd /home/ComfyUI
source /home/comfyui-venv/bin/activate

python main.py \
  --listen 0.0.0.0 \
  --port 6006 \
  --front-end-root /home/ComfyUI_frontend/dist
```

For a tracked Jarvis run from your local machine:

```bash
jl run --on <machine_id> --json --yes -- bash -lc \
  'cd /home/ComfyUI && source /home/comfyui-venv/bin/activate && exec python main.py --listen 0.0.0.0 --port 6006 --front-end-root /home/ComfyUI_frontend/dist'
```

Check logs:

```bash
jl run logs <run_id> --tail 80
```

Expected log line:

```text
To see the GUI go to: http://0.0.0.0:6006
```

## 4. Verify The Public Endpoint

Replace `YOUR_URL` with the first URL in `jl get <machine_id> --json`
`endpoints`.

```bash
curl -fsSI https://YOUR_URL/ | head
curl -fsS https://YOUR_URL/api/features
```

Expected feature flag:

```json
"jarvis_model_downloads": true
```

Verify that the Jarvis route exists:

```bash
curl -sS -X POST https://YOUR_URL/api/jarvis/models/download \
  -H 'Content-Type: application/json' \
  -d '{}' \
  -w '\nHTTP %{http_code}\n'
```

Expected:

```text
{"error": "Unsupported model download URL."}
HTTP 400
```

## 5. Optional: Verify A Real Remote Model Download

This test serves a tiny `.safetensors` file inside the container and asks
ComfyUI to download it into the normal `checkpoints` model directory. It proves
that downloads happen on the remote machine, not in the browser.

Start a local file server inside the container:

```bash
jl exec <machine_id> -- bash -lc '
set -euxo pipefail
mkdir -p /tmp/jarvis-model-source /home/ComfyUI/models/checkpoints
rm -f \
  /home/ComfyUI/models/checkpoints/jarvis_test.safetensors \
  /home/ComfyUI/models/checkpoints/jarvis_test.safetensors.part-* \
  /tmp/jarvis-http.log
printf jarvis-test-model > /tmp/jarvis-model-source/jarvis_test.safetensors
nohup /home/comfyui-venv/bin/python -m http.server 7010 \
  --bind 127.0.0.1 \
  --directory /tmp/jarvis-model-source \
  > /tmp/jarvis-http.log 2>&1 &
sleep 1
curl -fsS http://127.0.0.1:7010/jarvis_test.safetensors
'
```

Request a server-side model download:

```bash
curl -sS -X POST https://YOUR_URL/api/jarvis/models/download \
  -H 'Content-Type: application/json' \
  -d '{"url":"http://localhost:7010/jarvis_test.safetensors","filename":"jarvis_test.safetensors","directory":"checkpoints"}'
```

Expected response:

```json
{
  "status": "started",
  "task_id": "...",
  "filename": "jarvis_test.safetensors",
  "path": "/home/ComfyUI/models/checkpoints/jarvis_test.safetensors",
  "directory": "checkpoints"
}
```

Poll the task:

```bash
curl -sS https://YOUR_URL/api/jarvis/models/download/<task_id>
```

Expected final status:

```json
{
  "status": "completed",
  "bytes_downloaded": 17,
  "bytes_total": 17,
  "progress": 1.0,
  "error": null
}
```

Verify the file exists on the remote machine and no partial file remains:

```bash
jl exec <machine_id> -- bash -lc '
set -euxo pipefail
ls -l /home/ComfyUI/models/checkpoints/jarvis_test.safetensors
cat /home/ComfyUI/models/checkpoints/jarvis_test.safetensors
ls /home/ComfyUI/models/checkpoints/jarvis_test.safetensors.part-* 2>/dev/null || true
'
```

Expected file contents:

```text
jarvis-test-model
```

## 6. Open The UI

Open:

```text
https://YOUR_URL/
```

For a workflow with missing models, the Jarvis build should:

- Show the existing missing-model `Download`, `Download all`, `Copy URL`, and
  `Use from Library` controls.
- Download models into the remote ComfyUI model directory.
- Show progress in the missing-model status card.
- Not show a "download started" toast.
- Hide the download button after a model is completed locally.
