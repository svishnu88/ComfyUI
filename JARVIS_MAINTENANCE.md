# Jarvis Labs Maintenance Notes

This fork carries Jarvis Labs specific ComfyUI backend changes. Keep the patch
small, obvious, and gated so upstream updates remain easy to merge.

## Branches And Remotes

- `origin`: `https://github.com/svishnu88/ComfyUI.git`
- `upstream`: `https://github.com/comfyanonymous/ComfyUI.git`
- Jarvis branch: `jarvis-main`
- Upstream branch: `master`

## Jarvis Backend Surface

Jarvis-specific backend behavior should stay behind names that contain
`jarvis`:

- Server feature flag: `jarvis_model_downloads`
- API route prefix: `/api/jarvis/...`
- Current model download endpoints:
  - `POST /api/jarvis/models/download`
  - `GET /api/jarvis/models/download/{task_id}`

The model downloader intentionally:

- Accepts only workflow-provided URLs from the allowlisted model hosts.
- Saves models into the regular ComfyUI model directories via `folder_paths`.
- Never overwrites an existing completed model file.
- Downloads to `filename.part-<task_id>` and only renames after the stream
  completes.
- Removes stale partial files on a new download request.
- Limits concurrent downloads to `3`.
- Does not resume downloads after backend restart.

## Updating From Upstream

Use a merge-first workflow so Jarvis changes remain visible:

```bash
git checkout jarvis-main
git fetch upstream
git merge upstream/master
python -m py_compile server.py comfy_api/feature_flags.py
git diff --check
```

If conflicts happen, expect them around:

- `server.py`
- `comfy_api/feature_flags.py`

Preserve the `/jarvis/` route namespace and the `jarvis_model_downloads`
capability. Do not move this feature to a generic upstream-looking API unless
Jarvis and upstream intentionally agree to share the behavior.

## Deploying To Jarvis Labs

For a fresh PyTorch container setup, use the tested runbook in
`JARVIS_PYTORCH_CONTAINER_SETUP.md`.

Upload the changed backend files to the GPU instance, then restart ComfyUI:

```bash
jl upload <machine_id> server.py /home/ComfyUI/server.py --json
jl upload <machine_id> comfy_api/feature_flags.py /home/ComfyUI/comfy_api/feature_flags.py --json
jl run --on <machine_id> --json --yes -- bash -lc 'cd /home/ComfyUI && exec /home/comfyui-venv/bin/python main.py --listen 0.0.0.0 --port 6006 --front-end-root /home/ComfyUI_frontend/dist'
```

Verify:

```bash
curl -sS https://<jarvis-url>/api/features
curl -sS -X POST https://<jarvis-url>/api/jarvis/models/download \
  -H 'Content-Type: application/json' \
  -d '{}'
```

The empty POST should return HTTP 400 with an unsupported URL error.
