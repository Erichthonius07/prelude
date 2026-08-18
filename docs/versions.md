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
- **TensorFlow 2.20 deprecates `tf.lite`.** On-device conversion/inference has
  moved to a separate project, **LiteRT**, pinned here at **2.1.5** (the
  current latest release, verified directly against the project's GitHub
  releases page on 2026-08-17). Role 5a's quantization work and Role 5b's
  TFLite conversion smoke-test must target LiteRT's APIs, not the classic
  `tf.lite` converter — this directly affects the Prelude spec's stated
  deployment stack ("TensorFlow Lite + NNAPI delegate").
- `eclipse-temurin:25-jdk` / `25-jre` are used **without** the `-alpine`
  suffix — Alpine variants for a JDK release this recent (Java 25 GA'd
  September 2025) aren't reliably available across registries yet. Debian-based
  Temurin images are the safer default; revisit if image size becomes a
  problem.

## Known risks (not version bugs, but consequences of the version decisions above)

- **Restormer (Role 5b stretch goal) has no official TensorFlow/Keras
  checkpoint.** The official implementation and pretrained weights
  (swz30/Restormer) are PyTorch-only. Spec 5b.1 requires fine-tuning from a
  pretrained checkpoint, not training from scratch — on a TensorFlow-locked
  stack this needs one of: porting the PyTorch weights to TF/Keras, treating
  Restormer as a scoped one-off PyTorch exception, or accepting it as a
  candidate hard-blocker for the Phase 1-2 conversion smoke-test (which
  exists specifically to catch this kind of thing early — see
  `Role5b_Full_Spec.pdf` section 5b.2). **Not yet resolved — needs a team
  decision before Role 5b's Phase 1-2 work starts.**

## Resolved

- ~~PyTorch vs. TensorFlow~~ — decided 2026-08-18: **TensorFlow**. Locked in
  `training/requirements.txt`, no longer an open question.
