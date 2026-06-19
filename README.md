#

## Issue: PnP -> PIM bootstrap can fail on hard datasets

On harder datasets, the current PnP -> PIM bootstrap path can fail before the
pipeline switches to inertial prediction mode. PnP is used during the ramp-up
phase while the smoother is still trying to estimate the accelerometer bias. If
PnP loses enough visual correspondences or does not provide reliable pose
updates quickly enough, the accelerometer bias may not converge in time. In that
case the pipeline may never reach a stable PIM mode and the full VIO pipeline
will not work for that run.

## Prerequisites

- Python 3.13
- uv
- just

## Installation

```bash
brew install just
```
