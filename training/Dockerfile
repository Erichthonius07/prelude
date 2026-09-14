FROM python:3.13-slim

WORKDIR /workspace

# Force HTTPS for Debian's package mirrors — some networks block outbound
# HTTP (port 80) while allowing HTTPS (443), which is likely what's
# happening here.
RUN sed -i 's|http://deb.debian.org|https://deb.debian.org|g' /etc/apt/sources.list.d/debian.sources 2>/dev/null || true

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
