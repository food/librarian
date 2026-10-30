FROM python:3.12-slim

# git for bundle versioning (auto-commit per mutation)
RUN apt-get update && apt-get install -y --no-install-recommends git curl openssh-client \
    && rm -rf /var/lib/apt/lists/*

# Optional: pull pip packages through a local PyPI cache
ARG PIP_INDEX_URL=https://pypi.org/simple
ARG PIP_TRUSTED_HOST=
ENV PIP_INDEX_URL=${PIP_INDEX_URL} PIP_TRUSTED_HOST=${PIP_TRUSTED_HOST}

WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir .

COPY config.yaml ./config.yaml
COPY scripts ./scripts

# git identity for bundle commits
RUN git config --global user.name "librarian-bot" \
    && git config --global user.email "librarian@local" \
    && git config --global --add safe.directory /bundle

EXPOSE 8765
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -fsS http://localhost:8765/health || exit 1

CMD ["python", "-m", "librarian.server"]
