FROM node:22-bookworm-slim AS web-build

WORKDIR /src/webui
COPY webui/package.json webui/package-lock.json ./
RUN npm ci
COPY webui/ ./
RUN npm run build

FROM python:3.11-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    MYCLAW_WORKSPACE=/data \
    MYCLAW_REQUIRE_EXEC_SANDBOX=false

RUN apt-get update \
    && apt-get install --no-install-recommends -y ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY myclaw/ ./myclaw/
COPY --from=web-build /src/myclaw/web/dist/ ./myclaw/web/dist/
RUN python -m pip install --no-cache-dir .

RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin myclaw \
    && mkdir -p /data /workspace \
    && chown -R myclaw:myclaw /data /workspace

USER myclaw
WORKDIR /workspace

EXPOSE 8765
CMD ["python", "-m", "myclaw", "gateway", "--host", "0.0.0.0", "--port", "8765"]
