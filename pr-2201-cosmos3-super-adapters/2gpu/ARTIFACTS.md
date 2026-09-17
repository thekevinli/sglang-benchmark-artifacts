# Cosmos3-Super 2-H100 artifact index

This directory contains the curated evidence for the 2-H100 validation campaign.

- `PR_DESCRIPTION.md`: paste-ready validation write-up and generation examples.
- `generation-quality/`: recorded outputs for every generation mode — T2I image
  (1280x720), T2V/I2V/T2V+sound/V2V videos (1280x720x121), the forward-dynamics
  4-chunk rollout (640x640), the inverse-dynamics action JSON, plus contact
  sheets and ffprobe data.
- `smoke-*-layerwise.{log,xml}`, `generation-smoke-artifacts/`, and
  `t2i-restart-artifact/`: passing T2I and seven-case generation feature
  coverage. The T2I output is deterministic across both owner starts, so one of
  the two byte-identical images is retained while the log/XML records both runs.
- `reasoner-reference.{log,xml}` and `reasoner-reference-artifacts/`: passing
  text/image/video reasoner plumbing smoke cases with stable filenames.
- `reasoner-accuracy/`: MMMU/VideoMME accuracy run — `summary.json` plus
  raw per-sample `{mmmu,videomme}_results.json` (50 CI samples each).
- `lifecycle-reasoner.{log,xml}`: passing cancellation, failure, cleanup, and
  restart coverage.
- `expected-failure-*`: intentional evidence for the stock 2-GPU residency OOM
  and the released SGLang 0.5.19 reasoner-registration incompatibility. These
  are not current test regressions.
- `install_native_runtime.sh`: pins the native runtime. Everything else
  reproduces from committed tests under `tests/` (see PR_DESCRIPTION
  Reproduction). Weights download at serve time (`resolve_checkpoint`), so no
  separate model-download script.

`cosmos3-super-2gpu-artifacts.tar.gz` is a portable copy of the curated files.
`BUNDLE_SHA256` contains its SHA-256 integrity digest; it is not a benchmark,
model identifier, or quality score. Verify the archive before sharing with:

```bash
sha256sum -c BUNDLE_SHA256
```

After extraction, `SHA256SUMS` verifies every individual file in the archive:

```bash
sha256sum -c SHA256SUMS
```
