# Study server image. Python 3.10 matches the xaik-api-dev environment.
FROM python:3.10-slim

# libgomp is needed by xgboost/lightgbm-style wheels; the rest is build tooling
# a few scientific wheels still fall back to.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# typing-extensions first, from the normal index: every release proxied
# through the PyTorch CPU wheel index below has a metadata Name-casing
# mismatch ("typing_extensions" vs the requested "typing-extensions"), which
# makes pip discard the wheel and try to build the sdist instead -- needing
# flit_core, which that CPU-only index does not carry, failing the install no
# matter which version is requested. Satisfying the requirement here first
# means torch's own install below finds it already present and never touches
# that index for it.
RUN pip install --no-cache-dir "typing-extensions>=4.10,<5"

# Torch from the CPU wheel index: the default index serves the CUDA build,
# which is several GB larger and useless here -- the MLP trains fine on CPU.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        "torch>=2.5,<3"

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# .dockerignore keeps human participant data out of these trees -- see the
# "Human participant data" section there before widening any of them.
COPY src/ ./src/
COPY server/ ./server/
COPY assets/ ./assets/

ENV PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    XAIKIT_SERVER_RUNS_DIR=/data/server_runs

# Study artifacts belong on a volume so a redeploy does not discard finished runs.
VOLUME ["/data"]
EXPOSE 8000

# One worker on purpose: a study is a stateful in-process object, and stages run
# on a single background thread (see server/README.md).
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
