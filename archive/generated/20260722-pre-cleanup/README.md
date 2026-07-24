# Generated Artifact Archive

`historical-generated-artifacts.tar.gz` preserves all generated QC outputs that
existed before the 2026-07-22 cleanup except the retained canonical run
`interactive-review-final-20260722`. It also contains the retired repository-level
calibrated thresholds and shared logs.

`SHA256SUMS` verifies the compressed bundle. Paths in the archive are relative to
the `QC_System/` root and can be inspected with:

```bash
tar -tzf archive/generated/20260722-pre-cleanup/historical-generated-artifacts.tar.gz
sha256sum -c archive/generated/20260722-pre-cleanup/SHA256SUMS
```

The bundle is historical evidence, not an active pipeline input.
