FROM python:3.12-slim-bookworm AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 DATA_DIR=/tmp/vos-frame-studio
WORKDIR /app
COPY frame_studio/requirements.txt /app/frame_studio/requirements.txt
RUN --mount=type=secret,id=build_ca \
    set -eu; if [ -f /run/secrets/build_ca ]; then export PIP_CERT=/run/secrets/build_ca; fi; \
    apt-get update && apt-get install -y --no-install-recommends ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 vos && useradd --uid 10001 --gid vos --create-home vos \
    && pip install --only-binary=:all: -r /app/frame_studio/requirements.txt
COPY --chown=10001:10001 frame_studio /app/frame_studio
USER vos
EXPOSE 10000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s CMD python -c "import os,urllib.request;urllib.request.urlopen('http://127.0.0.1:'+os.getenv('PORT','10000')+'/healthz',timeout=4)"
CMD ["sh", "-c", "exec uvicorn frame_studio.app:app --host 0.0.0.0 --port ${PORT:-10000} --workers 1 --limit-concurrency 16 --timeout-keep-alive 10"]

FROM runtime AS test
COPY --chown=10001:10001 colab /app/colab
USER root
RUN --mount=type=secret,id=build_ca \
    set -eu; if [ -f /run/secrets/build_ca ]; then export PIP_CERT=/run/secrets/build_ca; fi; \
    pip install --only-binary=:all: -r /app/frame_studio/requirements-test.txt \
    && su vos -s /bin/sh -c 'python -m unittest discover -s frame_studio/tests -t . -v'
USER vos

FROM runtime AS production
