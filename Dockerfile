# ---- UI build ---------------------------------------------------------------
FROM node:22-slim AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
# Cypress/Storybook are dev tooling; skip the Cypress binary download in the image build.
RUN CYPRESS_INSTALL_BINARY=0 npm ci --no-audit --no-fund
COPY web/ ./
RUN npx tsc -b && npx vite build

# ---- engine + API -------------------------------------------------------------
FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY pyproject.toml README.md ./
COPY kairos/ kairos/
COPY benchmark/ benchmark/
COPY configs/ configs/
COPY corpus/ corpus/
COPY demo/ demo/
COPY models/ models/
COPY tests/ tests/
COPY --from=web /web/dist web/dist
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health')"
CMD ["python", "-m", "kairos", "serve", "--host", "0.0.0.0", "--port", "8000"]
