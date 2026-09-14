<div align="center">

# PRELUDE
### Burst Low-Light Denoising & Fusion Engine

A scoped, student-buildable realization of the architectural concepts behind Google's Night Sight and HDR+.  
Capture a burst in low light, align, fuse, denoise with a neural model, and post-process to produce a single clean photograph.

<br>

<img src="https://img.shields.io/badge/Java-25%20(LTS)-007396?style=for-the-badge&logo=java&logoColor=white" alt="Java 25">
<img src="https://img.shields.io/badge/Spring%20Boot-4.1.0-6DB33F?style=for-the-badge&logo=springboot&logoColor=white" alt="Spring Boot 4.1.0">
<img src="https://img.shields.io/badge/PostgreSQL-18--alpine-336791?style=for-the-badge&logo=postgresql&logoColor=white" alt="PostgreSQL 18">
<img src="https://img.shields.io/badge/TensorFlow-2.20.0-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow 2.20">
<img src="https://img.shields.io/badge/Data%20Contract-v1.0.0%20Ratified-success?style=for-the-badge" alt="Contract Ratified">

</div>

---

### The Pipeline

```text
[ Capture ] ➔ [ Align ] ➔ [ Fuse ] ➔ [ Denoise (AI) ] ➔ [ Post-process ] ➔ [ Final Image ]
```

> **Evaluation Model:** Distributed. Every stage reports metrics to the central Results Service.  
> **Contract:** `docs/data-contract.md` serves as the single source of truth for all inter-module communication.

---

### Repository Layout

| Directory | Owner | Purpose |
| :--- | :--- | :--- |
| **`results-service/`** | Role 1 | Spring Boot + PostgreSQL metric ingestion API (Dockerized, always-on). |
| **`capture-android/`** | Role 1 | Camera2 burst capture (Android Studio project, not Dockerized). |
| **`calibration-tool/`** | Role 1 | Deterministic 4-threshold on-device calibration. |
| **`batch-runner/`** | Role 1 | Headless 7-variant ablation runner. |
| **`alignment/`** | Role 2 | RANSAC alignment + learned confidence classifier. |
| **`fusion-postprocess-demo/`**| Role 3 | 4-way fusion, post-processing, demo-app UI. |
| **`training/`** | Roles 2-5b | Shared Python ML environment (Dockerized, on-demand). |
| **`quantization-deploy/`** | Role 5a | INT8 quantization, LiteRT/NNAPI deployment. |
| **`restormer-stretch/`** | Role 5b | Restormer conversion + fine-tuning + go/no-go. |
| **`docs/`** | All | Data contract, team map, versions, status. |

---

### Pinned Versions

| Component | Version | Notes |
| :--- | :--- | :--- |
| **Java** | `25` (LTS) | |
| **Spring Boot** | `4.1.0` | |
| **Maven** | `3.9.16` | |
| **PostgreSQL** | `18-alpine` | |
| **Flyway** | `12.4.0` | Boot-managed. PG 18 verified. |
| **Python** | `3.13` | |
| **ML Framework** | `TensorFlow 2.20.0` | Locked team decision (2026-08-18). |
| **LiteRT** | `2.1.5` | Required. TF 2.20 deprecates `tf.lite`. |

*Full rationale and compatibility notes live in `docs/versions.md`. Update there first if anything changes.*

---

### Getting Started

#### 1. Clone and configure
```bash
git clone <repository-url>
cd prelude
cp .env.example .env
# Edit .env with your local Postgres credentials
```

#### 2. Launch the always-on stack
```bash
docker compose up --build
```
This starts the **Results Service** and **PostgreSQL**.  
Verify the health check: `curl http://localhost:8080/` should return `{"status":"up"}`.  
*Note: This stack should run on one team-controlled laptop over local Wi-Fi, reachable by all Android devices on the same network.*

#### 3. Training and ML work (Roles 2–5b)
Use the separate, on-demand container defined in `training/README.md`. **Do not** add it to the root `docker-compose.yml`; it is resource-heavy and not required for all roles.

#### 4. Android Capture App
`capture-android/` is a standard Android Studio / Gradle project. It is not Dockerized, as it requires the Android SDK and a physical device or emulator.

---

### Data Layout

Bulk data lives **outside version control** under the repository root:

```text
data/
 ├── sidd/              # SIDD dataset
 ├── team-captures/     # Team-captured bursts
 ├── held-out/          # Locked held-out test set (see data-contract.md §8)
 ├── calibration/       # calibration-tool outputs
 └── checkpoints/       # Training checkpoints
```

The entire `data/` tree is **gitignored**. Datasets, captures, and checkpoints are never committed. Reference them strictly by `imageId` and SHA-256 hash as defined in the data contract.

---

### Branching Strategy

Single monorepo with feature branches per module (e.g., `role1/results-service`, `role2/alignment`).

**Strict Rule:** Any change to `docs/data-contract.md` requires Role 1's PR review. The contract is the single source of truth across all six modules; schema and payload drift is not permitted.

---

### Project Status

| Decision Area | Current State |
| :--- | :--- |
| **Data Contract** (`docs/data-contract.md` v1.0.0) | Ratified |
| **Flyway 12.4.0 ↔ PostgreSQL 18** | Verified |
| **Spring Boot 4 Web Starter** | Verified (`spring-boot-starter-webmvc`) |
| **Idempotency / Supersede Semantics** | Ratified & Implemented (Server-assigned versions) |
| **7-Variant Ablation List** | Ratified |
| **Test-Set Leakage Prevention** | Implemented (Triple-layer guard) |
| **Manifest Admin Auth** | Shared static token (Intentionally minimal) |
| **Read / Query API** | *Next Brief* (Demo app & bootstrap retrieval) |

*For granular module status and blockers, see `docs/STATUS.md`.*

---

### Quick Paths by Role

| Role | Domain | Next Action / Blocker |
| :--- | :--- | :--- |
| **Role 1** | `results-service`, `capture-android`, `calibration-tool`, `batch-runner` | Contract ratified. Next: Read-API brief. |
| **Role 2** | `alignment/` | Unblocked. See contract §6.2 (`AlignedFrameSet`) and §5.2. |
| **Role 3** | `fusion-postprocess-demo/` | Blocked on Role 2 output. See contract §6.3, §6.5, §5.3, §5.5. |
| **Role 4** | `training/` (Primary CNN) | Unblocked. See contract §8 (leakage guard) and §5.6. |
| **Role 5a** | `quantization-deploy/` | Blocked on Role 4 model. See `versions.md` for TF 2.20 deprecation notes. |
| **Role 5b** | `restormer-stretch/` | Unblocked for Phase 1–2 smoke test. See contract §5.4 (schema-enforced anchoring). |
