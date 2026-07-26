#

## Issue: PnP -> PIM bootstrap can fail on hard datasets

On harder datasets, the current PnP -> PIM bootstrap path can fail before the
pipeline switches to inertial prediction mode. PnP is used during the ramp-up
phase while the smoother is still trying to estimate the accelerometer bias. If
PnP loses enough visual correspondences or does not provide reliable pose
updates quickly enough, the accelerometer bias may not converge in time. In that
case the pipeline may never reach a stable PIM mode and the full VIO pipeline
will not work for that run.

## Issue: Frame-to-frame PnP degrades at very low stereo ratio

The current frame-to-frame PnP path is fragile when `stereo_ok_ratio` becomes
very low. Even if the left-image tracker still keeps many features alive, the
PnP store is seeded from stereo-triangulated points, so long stretches with only
a few valid stereo matches can leave too few reliable 3D correspondences for the
next pose estimate. In this mode PnP can either fail or produce a corrupted pose,
especially in close-range scenes where stereo geometry is sparse or unstable.

This is a known limitation of the current frontend VO design. The planned
direction is to move toward observation-history based landmark initialization,
where tracked mono observations can accumulate parallax and be triangulated with
multi-view GN instead of depending only on per-frame stereo depth.

## Prerequisites

- Python 3.13
- uv
- just

## Installation

```bash
brew install just
```
