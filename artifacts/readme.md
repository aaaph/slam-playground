# Artifact Manifests

This directory stores small YAML manifests for research artifacts: visual vocabularies, neural network
weights, learned indices, and dataset-like payloads that are not regular sensor datasets.

Commit manifests here, but keep the actual artifact payloads local or fetched from their source. Store payloads
under per-artifact directories such as `artifacts/orb_vocabulary/`. The repository `.gitignore` keeps large
payload files under `artifacts/` out of git by default while allowing top-level `*.yaml` and `*.yml` manifests.

Common commands:

```bash
just artifact list
just artifact show orb_vocabulary
just artifact path orb_vocabulary dbow3
just artifact fetch orb_vocabulary
just artifact verify orb_vocabulary
```
