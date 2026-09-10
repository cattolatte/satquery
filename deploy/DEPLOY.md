# Publishing SatQuery to Hugging Face Spaces

Start to end. Produces a public URL of the form
`https://huggingface.co/spaces/<username>/satquery` that anyone can open, which
is what the SIH portal wants.

---

## Read this first: Docker Spaces is now paid

The Space creation page shows **Docker** with a padlock and a *Paid* badge. Only
**Static** and **Gradio** are on the free tier, so a Python app must be Gradio.

That turns out to be the better path anyway, because **ZeroGPU is Gradio-only**
— so the same file that runs free on CPU can also run free on a GPU.

`deploy/app.py` is that Gradio front end. It is an interface layer over the
existing `Controller.run(query, image_paths)`: routing, tool selection, fusion,
confidence and the trace are untouched, so the Space demonstrates the same code
path the benchmarks were measured on.

## Which hardware to pick

| | CPU Basic | ZeroGPU |
|---|---|---|
| Cost | free | free |
| Specs | 2 vCPU, 16 GB RAM | RTX Pro 6000 Blackwell, 48 GB VRAM |
| Speed | several seconds per query | fast |
| Quota | unlimited | **5 min GPU/day** on a free account |
| Eligibility | anyone | account **> 30 days old**, verified email, max 2 |

**Pick ZeroGPU if your account is older than 30 days.** `app.py` already carries
the `@spaces.GPU` decorator and no-ops it when the `spaces` module is absent, so
the same file works on either. If the account is too new, ZeroGPU will not be
selectable — use CPU Basic and expect several seconds per query.

---

## What actually gets deployed

Not the whole repository, and not 2.9 GB of checkpoints.

| Part | Size | How it arrives |
|---|---|---|
| `checkpoints/rs_clip` | 581 MB | pulled from a model repo at startup |
| `checkpoints/rs_vlm` | 19 MB | pulled from a model repo at startup |
| detector base (`owlv2-base-patch16-ensemble`) | ~600 MB | downloaded by `transformers` on first load |
| generative base | — | downloaded on first load |
| `satquery/` + `app.py` | small | pushed with the Space |

The other eight checkpoint directories — `rs_clip_openai`, `rs_vlm_v1_random`
and the rest — are ablation artefacts and are **not** needed to serve.

---

## Step 1 — Upload the two serving adapters

Once. Keeps 600 MB out of the Space's git history and makes startup a
Hub-to-Hub fetch rather than an upload from your laptop.

```bash
pip install -U "huggingface_hub[cli]"
hf auth login          # a WRITE token from huggingface.co/settings/tokens
```

```bash
cd ~/Workspace/satquery
hf repo create satquery-weights --repo-type model
hf upload satquery-weights checkpoints/rs_clip  rs_clip  --repo-type model
hf upload satquery-weights checkpoints/rs_vlm   rs_vlm   --repo-type model
```

The first upload is ~581 MB and is the slow part. Start it and do something
else.

---

## Step 2 — Create the Space

**New → Space** on huggingface.co.

| Field | Value |
|---|---|
| Space name | `satquery` |
| License | MIT |
| SDK | **Gradio** → template **Blank** |
| Hardware | **ZeroGPU** if offered, else **CPU basic** |
| Visibility | **Public** |

It must be **public** for a judge to open it without logging in.

---

## Step 3 — Assemble the Space repository

```bash
git clone https://huggingface.co/spaces/<username>/satquery hf-satquery
cd hf-satquery

cp -r ~/Workspace/satquery/satquery ./satquery      # the system itself
cp ~/Workspace/satquery/deploy/app.py            ./app.py
cp ~/Workspace/satquery/deploy/requirements.txt  ./requirements.txt
cp ~/Workspace/satquery/deploy/README-space.md   ./README.md
```

**`README.md` must be the one from `deploy/`.** Its YAML front matter declares
`sdk: gradio` and `app_file: app.py`. Without it the Space fails to build with
a confusing *"No application file"* error.

Keep the bulk out:

```bash
printf 'checkpoints/\ndata/\neval/\ntests/\nweb/\n__pycache__/\n*.pyc\n' > .gitignore
```

---

## Step 4 — Point it at the weights

**Settings → Variables and secrets → New variable**

```
Name   WEIGHTS_REPO
Value  <username>/satquery-weights
```

A **variable**, not a secret — it is not sensitive.

If it is unset the Space still runs, unadapted, and **caps its own confidence**,
which is the designed behaviour rather than a silent downgrade.

---

## Step 5 — Push

```bash
git add -A
git commit -m "Deploy SatQuery AI"
git push
```

Build takes roughly **10–20 minutes**; installing PyTorch is most of it. Watch
**Logs → Build**.

---

## Step 6 — Verify before submitting the URL

- [ ] Space shows **Running** (green)
- [ ] The interface renders and the example queries populate the box
- [ ] One real query returns an answer **with evidence and a trace**
- [ ] Confidence is not pinned near zero — if it is, `WEIGHTS_REPO` is wrong
- [ ] Opened in a **private window** — proves it works without your login

**Warm it before a demo.** Free Spaces sleep after ~48 hours idle and take a
minute to wake, and the first query is slower again while `transformers`
downloads the detector base.

---

## If it fails

| Symptom | Cause |
|---|---|
| `No application file` | `README.md` missing its front matter, or `app_file` is not `app.py` |
| `ModuleNotFoundError: satquery` | the `satquery/` package was not copied into the Space root |
| Build times out | PyTorch download — retry; the layer cache survives |
| Confidence always low | `WEIGHTS_REPO` unset or misspelled; the backbone is unadapted |
| ZeroGPU not selectable | account under 30 days old, or email unverified |
| `spaces` import error locally | expected — `app.py` no-ops the decorator off-Hub |
| Killed during load | 16 GB exceeded; check no experiment checkpoints were copied in |

---

## The honest framing for the submission

On CPU this is a **CPU demo of a system measured on GPU**. Say so. The benchmark
numbers in the Space README were produced through the real serving path on
appropriate hardware; the Space exists so a reviewer can click through the
interface and read the trace, not to reproduce timings.
