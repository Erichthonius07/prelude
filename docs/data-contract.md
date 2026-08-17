# Data Contract

**Status: not yet written.** Owned by Role 1 (you). Must be published before
Phase 2 ends — every other role builds against it.

Per the Prelude (4.1) and Role 1 spec (2.2, 2.6), this needs to define, at minimum:

- Field names/types for metric submissions from each stage (Fusion's 4-strategy
  scores, Denoise's deployment metrics, PostProcess's SSIM contribution, etc.)
- Per-image score requirement (not just aggregates) — needed for bootstrap resampling
- Idempotency key strategy (safe against duplicate/retried submissions)
- Schema versioning approach and the add-only evolution policy
- The FrameBurst / AlignedFrameSet / FusedFrame / DenoisedFrame / FinalImage
  contracts between pipeline stages themselves (not just the Results Service API)

This is the next thing to build after this scaffold — flag it back to Claude
when you're ready and we'll design it properly.
