#!/usr/bin/env bash
# Create annotated tags for the major project stages and push them.
# (Tag pushes were blocked from the development environment, so run this from a clone.)
# Usage: scripts/tag_stages.sh [remote]   (default remote: origin)
set -euo pipefail
remote="${1:-origin}"
git fetch "$remote" main

tag() {  # tag <name> <commit> <message>
  if git rev-parse -q --verify "refs/tags/$1" >/dev/null; then
    echo "tag $1 exists, skipping"
  else
    git tag -a "$1" "$2" -m "$3"
    echo "tagged $1 -> $2"
  fi
}

tag v0.1-setup       adcc6de "Stage 1: project setup (config, resource profiles, AOI, grids)"
tag v0.2-data        0d95ffd "Stage 2: data loaders (lidar, AlphaEarth, DEM, HLS, AVIRIS-3, references); S1 loader follows in cf5c9e2"
tag v0.3-labels      408f1d7 "Stage 3: lidar QC, stability screening, sensor harmonization"
tag v0.4-cv-baseline ae041bb "Stage 4: CV framework and AlphaEarth baseline"
tag v0.5-model       5016fbf "Stage 5: ablations, uncertainty calibration, temporal smoothing; final configuration v1_aef20"
tag v1.0-product     dda0b48 "Stage 6: final 20 m product 2017-2025 with calibrated uncertainty"
tag v1.0-site        2b36d93 "Stage 7: static website and web COGs"
tag v2.0-ssm         9d7e33d "Stage 8: monthly state-space model (evaluated against v1; site updated)"

git push "$remote" --tags
