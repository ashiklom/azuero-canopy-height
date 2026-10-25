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

git push "$remote" --tags
