# Cosmos3-Super 2-H100 artifact index

This directory contains the curated evidence for the 2-H100 validation campaign.

- `PR_DESCRIPTION.md`: paste-ready validation write-up and generation examples.
- `generation-quality/`: final 1280x720x189 T2V, I2V, and audiovisual outputs,
  structured prompt inputs, contact sheets, compact summaries, and ffprobe data.
- `smoke-*-layerwise.{log,xml}`, `generation-smoke-artifacts/`, and
  `t2i-restart-artifact/`: passing T2I and seven-case generation feature
  coverage. The T2I output is deterministic across both owner starts, so one of
  the two byte-identical images is retained while the log/XML records both runs.
- `reasoner-reference.{log,xml}` and `reasoner-reference-artifacts/`: passing
  text/image/video reference comparisons with stable, descriptive filenames.
- `lifecycle-reasoner.{log,xml}`: passing cancellation, failure, cleanup, and
  restart coverage.
- `expected-failure-*`: intentional evidence for the stock 2-GPU residency OOM
  and the released SGLang 0.5.19 reasoner-registration incompatibility. These
  are not current test regressions.
- `download_model.sh`, `install_native_runtime.sh`, `run_h100_validation.sh`, and
  `run_generation_quality.sh`: token-free reproduction entry points.

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
