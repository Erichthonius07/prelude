# Pinned Versions

Locked by the team on 2026-08-17. Change here first, then propagate to the
actual config files listed in the right column.

| Component | Version | Where it's pinned |
|---|---:|---|
| Java | 25 (LTS) | `results-service/pom.xml` (`java.version`), `results-service/Dockerfile` (base images) |
| Spring Boot | 4.1.0 | `results-service/pom.xml` (parent) |
| Maven | 3.9.16 | `results-service/Dockerfile` (downloaded explicitly from Apache archive) |
| PostgreSQL | 18-alpine | `docker-compose.yml` |
| Python | 3.13 | `training/Dockerfile` |
| PyTorch | 2.12.1 | `training/requirements.txt` (commented, pick one framework) |
| TensorFlow | 2.20.0 | `training/requirements.txt` (commented, pick one framework) |

## Known compatibility notes (verified 2026-08-17)

- **Spring Boot 4.1.0** requires Java 17+ and supports up to Java 26 — Java 25 is a valid, current choice.
- **PostgreSQL 18** changed its Docker image conventions: the `VOLUME` moved
  from `/var/lib/postgresql/data` to `/var/lib/postgresql`, and `PGDATA` is
  now version-namespaced internally (`/var/lib/postgresql/18/docker`).
  `docker-compose.yml` mounts the parent directory accordingly — don't revert
  this if you copy patterns from older Postgres tutorials.
- **TensorFlow 2.20 deprecates `tf.lite`.** On-device conversion/inference is
  moving to a separate project, **LiteRT**. If the team locks TensorFlow as
  the training framework, **Role 5a's quantization work and Role 5b's TFLite
  conversion smoke-test need to target LiteRT's APIs**, not the classic
  `tf.lite` converter — this directly affects the Prelude spec's stated
  deployment stack ("TensorFlow Lite + NNAPI delegate"). Raise this with the
  team before the framework decision is finalized, not after Role 4 has
  already trained against one path.
- No compatibility issue found between PyTorch 2.12.1 and Python 3.13.
- `eclipse-temurin:25-jdk` / `25-jre` are used **without** the `-alpine`
  suffix — Alpine variants for a JDK release this recent (Java 25 GA'd
  September 2025) aren't reliably available across registries yet. Debian-based
  Temurin images are the safer default; revisit if image size becomes a
  problem.

## Still open
- **PyTorch vs. TensorFlow** — team decision not yet made (Prelude 2.1). Given
  the `tf.lite`/LiteRT churn above, this is worth deciding sooner rather than
  later; it affects Role 5a and 5b's actual implementation path, not just a
  training-time choice.
- Exact `torchvision` version to pair with `torch==2.12.1` — not independently
  verified, marked in `requirements.txt` to double-check before locking.
