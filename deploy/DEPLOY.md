# Publishing SatQuery to Hugging Face Spaces

Start to end. Produces a public URL of the form
`https://huggingface.co/spaces/<username>/satquery` that anyone can open, which
is what the SIH portal wants.

**Free tier: CPU Basic — 2 vCPU, 16 GB RAM, no time limit, no card.**

---

## Why CPU Basic and not the free GPU

ZeroGPU is genuinely free (48 GB VRAM) but is **Gradio-only**, and this app is
FastAPI serving its own interface. Using it would mean rewriting the front end
for a demo, and a free account gets 5 minutes of GPU per day.

CPU Basic runs the existing app unchanged. The trade is speed: **several seconds
per query instead of sub-second.** That is acceptable for someone clicking
through a handful of examples, and it does not change any benchmark number —
those were measured on GPU and are reported as such.

---

## What actually gets deployed

Not the whole repository, and not 2.9 GB of checkpoints.

| Part | Size | How it arrives |
|---|---|---|
| `checkpoints/rs_clip` | 581 MB | pulled from the Hub at build time |
| `checkpoints/rs_vlm` | 19 MB | pulled from the Hub at build time |
| detector base (`owlv2-base-patch16-ensemble`) | ~600 MB | downloaded by `transformers` on first load |
| generative base | — | downloaded on first load |
| application code + `web/` | small | pushed with the Space |

The other eight checkpoint directories are experiment variants — `rs_clip_openai`,
`rs_vlm_v1_random` and so on. They are used by the ablations and are **not**
needed to serve.

---

## Step 1 — Upload the two serving adapters to a model repo

Do this once. It keeps 600 MB out of the Space's git history and makes the
build a Hub-to-Hub fetch rather than an upload from your laptop.

```bash
pip install -U "huggingface_hub[cli]"
hf auth login          # paste a WRITE token from huggingface.co/settings/tokens
```

```bash
cd ~/Workspace/satquery
hf repo create satquery-weights --repo-type model
hf upload satquery-weights checkpoints/rs_clip  rs_clip  --repo-type model
hf upload satquery-weights checkpoints/rs_vlm   rs_vlm   --repo-type model
```

The first upload is ~581 MB and is the slow part. Do it on good wifi.

---

## Step 2 — Create the Space

On huggingface.co: **New → Space**.

| Field | Value |
|---|---|
| Space name | `satquery` |
| License | MIT |
| SDK | **Docker** (blank template) |
| Hardware | **CPU basic — free** |
| Visibility | **Public** |

It must be **public** for the URL to be openable by judges without a login.

---

## Step 3 — Assemble the Space repository

The Space is its own git repo. Copy in the application, not the checkpoints.

```bash
git clone https://huggingface.co/spaces/<username>/satquery hf-satquery
cd hf-satquery

# application code and the interface it serves
cp -r ~/Workspace/satquery/satquery ./satquery
cp -r ~/Workspace/satquery/web      ./web

# deployment files
cp ~/Workspace/satquery/deploy/Dockerfile        ./Dockerfile
cp ~/Workspace/satquery/deploy/requirements.txt  ./requirements.txt
cp ~/Workspace/satquery/deploy/README-space.md   ./README.md
```

**`README.md` must be the one from `deploy/`** — its YAML front matter is what
tells Spaces to use Docker and port 7860. Without it the Space will not build.

Keep the checkpoints out:

```bash
printf 'checkpoints/\ndata/\neval/\ntests/\n__pycache__/\n*.pyc\n' > .gitignore
```

---

## Step 4 — Point the build at the weights

In the Space: **Settings → Variables and secrets → New variable**

```
Name   WEIGHTS_REPO
Value  <username>/satquery-weights
```

A **variable**, not a secret — it is not sensitive and the Dockerfile needs it
at build time.

If you skip this the Space still builds and runs, but unadapted — and it will
cap its own confidence, which is the designed behaviour rather than a silent
downgrade.

---

## Step 5 — Push

```bash
git add -A
git commit -m "Deploy SatQuery AI"
git push
```

The build takes roughly 10–20 minutes: installing PyTorch is most of it.
Watch **Logs → Build** in the Space.

---

## Step 6 — Verify before you submit the URL

- [ ] Space shows **Running** (green)
- [ ] The page loads and the interface renders
- [ ] `GET /api/health` reports the BigEarthNet fine-tune, not stock CLIP
- [ ] One real query returns an answer with evidence and a trace
- [ ] Opened in a private window — confirms it works without your login

**Warm it before a demo.** Free Spaces sleep after ~48 hours idle and take a
minute or so to wake. The very first query is slower again while
`transformers` downloads the detector base.

---

## If the build fails

| Symptom | Cause |
|---|---|
| `No application file` | `README.md` is missing its YAML front matter, or `sdk` is not `docker` |
| Build times out | PyTorch download; retry — the layer cache survives |
| `Permission denied` writing cache | `HF_HOME` not writable; the Dockerfile sets it under `/home/user` |
| Health reports stock CLIP | `WEIGHTS_REPO` unset or misspelled |
| Port error | Spaces requires 7860; `app_port` in the front matter must match the `CMD` |
| Killed during load | 16 GB exceeded — check no experiment checkpoints were copied in |

---

## The honest framing for the submission

This is a **CPU demo of a system measured on GPU**. Say so. The benchmark
numbers in the README were produced through the real serving path on
appropriate hardware; the Space exists so a reviewer can click through the
interface and see the trace, not to reproduce the timings.
