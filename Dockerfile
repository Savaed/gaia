FROM python:3.11.5-bullseye as builder

ARG POETRY_VERSIONS=1.6.1

RUN pip install poetry==${POETRY_VERSIONS}

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

WORKDIR /app

COPY pyproject.toml poetry.lock ./
RUN touch README.md
RUN --mount=type=cache,target=${POETRY_CACHE_DIR} poetry install --only main --no-root

FROM python:3.11.5-slim-bullseye as runtime

ARG VIRTUAL_ENVIRONMENT=/app/.venv

ENV VIRTUAL_ENV=${VIRTUAL_ENVIRONMENT} \
    PYSPARK_PYTHON=${VIRTUAL_ENVIRONMENT}/bin/python \
    PATH=${VIRTUAL_ENVIRONMENT}/bin:${PATH} \
    PYTHONPATH=/app

# (Required) Install packages spark uses to run
RUN apt update && apt install --yes procps tini

COPY --from=builder ${VIRTUAL_ENVIRONMENT} ${VIRTUAL_ENVIRONMENT}

WORKDIR /app
COPY gaia ./gaia

# (Required) Create the 'spark' group/user.
# The GID and UID must be 1099. Home directory is required.
RUN groupadd --gid  1099 spark
RUN useradd --uid 1099 --gid 1099 --home-dir /home/spark --create-home spark
USER spark
