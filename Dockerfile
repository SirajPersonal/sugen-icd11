# ICD-11 coder. Build: docker compose build   Run: docker compose run coder /in/notes.jsonl /out/predictions.jsonl
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies first (cached layer). Build time may use the network; run time cannot.
COPY requirements.txt .
RUN pip install -r requirements.txt

# Code and data. data/ holds the MMS release and, if fetched, WHO index terms.
COPY coder/ coder/
COPY data/ data/

# Build the retrieval index now, so the run does no network and no heavy work.
RUN python -m coder.index && python -c "import coder.index as i; i.load()"

ENTRYPOINT ["python", "-m", "coder"]
