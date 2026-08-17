# training/

Shared Python/ML environment for **Role 2** (alignment classifier), **Role 3**
(learned fusion model), **Role 4** (primary CNN), **Role 5a** (quantization),
and **Role 5b** (Restormer fine-tuning).

This is intentionally **not** part of `docker-compose.yml`. It's heavy,
possibly GPU-dependent, and only needed on-demand — not something every
teammate should have spinning up whenever they run the Results Service stack.

## Build
```bash
docker build -t prelude-training .
```

## Run (mount your code + data, don't bake them into the image)
```bash
docker run --rm -it \
  -v $(pwd)/../training:/workspace/training \
  -v $(pwd)/../data:/workspace/data \
  prelude-training
```

Add `--gpus all` if running on a machine with an NVIDIA GPU + nvidia-container-toolkit installed.

## Before you start
The team must lock **PyTorch vs. TensorFlow once** (Prelude, Section 2.1) —
uncomment the right block in `requirements.txt` and rebuild. This affects
Role 5a's TFLite conversion path and the Results Service schema, so it's a
one-time team decision, not per-role.
