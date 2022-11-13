FROM python:3.9

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_NO_INTERACTION=1 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VERSION=1.2.2

ENV PATH="$POETRY_HOME/bin:$PATH"

# Install poetry
RUN curl -sSL https://install.python-poetry.org | POETRY_VERSION=$POETRY_VERSION python3 -

WORKDIR /quantpy
COPY pyproject.toml poetry.lock  ./

# Install dependencies and package
RUN mkdir quantpy && \
    touch quantpy/__init__.py && \
    poetry config virtualenvs.create false && \
    poetry config installer.parallel true && \
    poetry install --no-interaction --no-ansi

# Copy code
COPY . .

ENTRYPOINT [ "poetry" , "run" ]
