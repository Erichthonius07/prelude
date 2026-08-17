FROM python:3.13-slim

WORKDIR /workspace

# System deps OpenCV needs at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Code and data are bind-mounted at run time, not baked into the image —
# checkpoints/datasets are large and gitignored, don't COPY them in.
CMD ["bash"]
