# Running SatQuery AI

## Install

```bash
pip install torch transformers open_clip_torch fastapi uvicorn pillow numpy pandas rasterio pytest
```

`rasterio` is only needed for GeoTIFF input; everything else runs without it,
and the registry reports the affected tool as unavailable rather than failing.

## Start the web application

```bash
cd satquery && PYTHONPATH=. python3 -m uvicorn satquery.server:app --port 8077
```

Open http://127.0.0.1:8077. Drop one or two images, type a question, press Ask.

The first request loads the backbone and takes a few seconds; later ones are
fast. `GET /api/health` reports which weights are live — expect
`RemoteCLIP -> BigEarthNet.txt fine-tune` once the checkpoint exists.

## Reproduce the adaptation

Prepare the pairs from BigEarthNet.txt, then fine-tune:

```bash
PYTHONPATH=. python3 satquery/adapt/prepare.py
```

```bash
PYTHONPATH=. python3 satquery/adapt/train.py --base remoteclip --epochs 6
```

Roughly 6 minutes on an M-series GPU. It prints before/after on the held-out
patches and writes `checkpoints/rs_clip/`, which the backbone picks up
automatically on next load. Pass `--base openai` to start from stock CLIP
instead; RemoteCLIP wins, but the comparison is one flag.

## Tests

```bash
PYTHONPATH=. python3 -m pytest tests/ -q
```

49 tests, about 3 seconds, no network and no weights required.

## API

| Endpoint | Purpose |
|---|---|
| `POST /api/query` | `q` plus one or two `images`; returns answer, confidence, evidence, trace |
| `GET /api/tools` | the registry, including what is currently unavailable and why |
| `GET /api/report/{id}.html` | self-contained report, images embedded |
| `GET /api/report/{id}.json` | same payload, machine-readable |
| `GET /api/health` | which backbone and device are live |

```bash
curl -X POST http://127.0.0.1:8077/api/query \
  -F "q=Highlight the water body referred to in the query." \
  -F "images=@path/to/scene.tif"
```

## Fetching the benchmarks

CDVQA and RSVQA-LR are already reproducible from small downloads. VRSBench VQA
and captioning need a 4 GB image archive that HuggingFace serves slowly to
unauthenticated clients — it stalls repeatedly on a weak connection.

Two things make it reliable:

**Use `curl -C -`, not the HuggingFace client.** The client hangs on a dropped
connection without raising, so its retry loop never fires. `curl` resumes from
the byte it stopped at, and rerunning the loop below picks up each time.

```bash
bash scripts/fetch_vrsbench.sh
```

**Set an `HF_TOKEN` if you have one.** HuggingFace throttles unauthenticated
transfers, which is the underlying cause. A free read token removes the cap.

```bash
export HF_TOKEN=hf_yourtoken
```

Once `data/bench/vrsbench/Images_val.zip` is extracted:

```bash
PYTHONPATH=. python3 eval/vrsbench_grounding.py --tiles 5 --min-area 1 --threshold 0.75
```
