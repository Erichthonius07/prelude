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
| ML framework | **TensorFlow 2.20.0** (locked, team decision) | `training/requirements.txt` |
| LiteRT | 2.1.5 (required — TF 2.20 deprecates `tf.lite`) | `training/requirements.txt` |

## Known compatibility notes (verified 2026-08-17)

- **Spring Boot 4.1.0** requires Java 17+ and supports up to Java 26 — Java 25 is a valid, current choice.
- **PostgreSQL 18** changed its Docker image conventions: the `VOLUME` moved
  from `/var/lib/postgresql/data` to `/var/lib/postgresql`, and `PGDATA` is
  now version-namespaced internally (`/var/lib/postgresql/18/docker`).
  `docker-compose.yml` mounts the parent directory accordingly — don't revert
  this if you copy patterns from older Postgres tutorials.
- **TensorFlow 2.20 deprecates `tf.lite`.** On-device conversion/inference is
  moving to a separate project, **LiteRT** (pinned here at **2.1.5**, the
  current latest release as of 2026-08-17 — note the person who first
  suggested this said 2.1.6, which does not exist; verified the real latest
  tag directly against the project's releases page before pinning). If the
  team locks TensorFlow as the training framework, **Role 5a's quantization
  work and Role 5b's TFLite conversion smoke-test need to target LiteRT's
  APIs**, not the classic `tf.lite` converter — this directly affects the
  Prelude spec's stated deployment stack ("TensorFlow Lite + NNAPI delegate").
  Raise this with the team before the framework decision is finalized, not
  after Role 4 has already trained against one path.
- No compatibility issue found between PyTorch 2.12.1 and Python 3.13.
- `eclipse-temurin:25-jdk` / `25-jre` are used **without** the `-alpine`
  suffix — Alpine variants for a JDK release this recent (Java 25 GA'd
  September 2025) aren't reliably available across registries yet. Debian-based
  Temurin images are the safer default; revisit if image size becomes a
  problem.

## Still open
- ~~PyTorch vs. TensorFlow~~ — **Decided 2026-08-18: TensorFlow.** Locked in
  `training/requirements.txt`.
- **New risk surfaced by the TensorFlow decision**: Restormer's official
  implementation and released pretrained checkpoints (Role 5b's stretch goal)
  are PyTorch-only. There's no official TF/Keras port. Since 5b.1 requires
  fine-tuning from a pretrained checkpoint rather than training from scratch,
  this needs a resolution before Role 5b's Phase 1-2 work — either port the
  PyTorch weights to TF, treat Restormer training as a scoped one-off PyTorch
  exception, or let the already-scheduled Phase 1-2 conversion smoke-test
  catch it as a hard blocker (which it's designed to do). See
  `training/requirements.txt` for the full note.
