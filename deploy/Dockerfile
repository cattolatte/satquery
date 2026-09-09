# Hugging Face Spaces — Docker SDK.
#
# Spaces requires the app to listen on 7860 and to run as a non-root user with
# UID 1000, which is why the user is created explicitly rather than relying on
# the base image.
#
# The two adapters this serves with -- checkpoints/rs_clip (581 MB) and
# checkpoints/rs_vlm (19 MB) -- are pulled from the Hub at build time rather
# than committed to the Space, because a 600 MB git push from a laptop is slow
# and fragile where a Hub-to-Hub fetch is neither. The detector and generative
# base models are downloaded by transformers on first load.
FROM python:3.11-slim

# rasterio needs GDAL; libgl and libglib are the usual OpenCV/Pillow runtime deps.
RUN apt-get update && apt-get install -y --no-install-recommends \
        gdal-bin libgdal-dev libgl1 libglib2.0-0 git \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONPATH=/home/user/app \
    PYTHONUNBUFFERED=1 \
    # Model cache must live somewhere the non-root user can write.
    HF_HOME=/home/user/.cache/huggingface \
    # CPU Basic has 2 vCPU; letting torch spawn more threads than that makes
    # inference slower, not faster.
    OMP_NUM_THREADS=2 \
    TOKENIZERS_PARALLELISM=false

WORKDIR $HOME/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .

# Fetch the two serving adapters. WEIGHTS_REPO is the model repo holding
# rs_clip/ and rs_vlm/; set it as a Space variable. If it is unset the build
# still succeeds and the app starts unadapted -- and caps its own confidence,
# which is the designed behaviour rather than a silent downgrade.
ARG WEIGHTS_REPO=""
RUN if [ -n "$WEIGHTS_REPO" ]; then \
      python -c "\
import os;\
from huggingface_hub import snapshot_download;\
snapshot_download(os.environ['WEIGHTS_REPO'], repo_type='model',\
                  local_dir='checkpoints', allow_patterns=['rs_clip/*','rs_vlm/*'])" ; \
    else echo 'WEIGHTS_REPO unset - starting unadapted'; fi
ENV WEIGHTS_REPO=${WEIGHTS_REPO}

EXPOSE 7860
CMD ["uvicorn", "satquery.server:app", "--host", "0.0.0.0", "--port", "7860"]
