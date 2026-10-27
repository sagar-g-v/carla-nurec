#!/usr/bin/env bash
# NVIDIA NuRec reconstruction of the CARLA recording, for a Linux x86_64 GPU machine (NOT runnable on the
# Windows laptop this repo was made on). Commands follow NVIDIA's NuRec 26.04 docs:
#   https://docs.nvidia.com/nurec/nurec/nurec-aux-data.html
#   https://docs.nvidia.com/nurec/nurec/reconstruct-av-scene.html
# Requirements (docs.nvidia.com/nurec/basics/hardware.html): Ampere or newer GPU with >= 24 GB VRAM (48 GB
# recommended, e.g. L40S / A6000 / A100 / H100), driver R550+ (R570 recommended), Docker >= 23.0.1,
# NVIDIA Container Toolkit >= 1.13.5, ~100 GB free disk, an NGC API key (free NGC account).
# Usage: NGC_API_KEY=... ./run_nurec.sh /path/to/ncore_dir /path/to/output [epochs]
#   ncore_dir: the output of convert/to_ncore.py (sequence .json manifest + .zarr.itar stores)
set -euo pipefail
: "${NGC_API_KEY:?set NGC_API_KEY (https://ngc.nvidia.com -> Setup -> API key)}"
DATA=$(realpath "${1:?ncore dir}"); OUT=$(realpath -m "${2:?output dir}"); EPOCHS=${3:-30}
TAG=26.04
SEQ=$(basename "$(ls "$DATA"/*.json | grep -v '\.aux\.' | head -1)")
CAMS="camera_front,camera_front_left,camera_front_right,camera_back_left,camera_back_right,camera_back"
mkdir -p "$OUT/aux" "$OUT/train"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo "$NGC_API_KEY" | docker login nvcr.io -u '$oauthtoken' --password-stdin
docker pull nvcr.io/nvidia/nre/nre-tools-ga:$TAG
docker pull nvcr.io/nvidia/nre/nre-ga:$TAG

# 1 auxiliary data (Mask2Former segmentation, lidar segmentation / camera visibility, metadata)
docker run --shm-size=2g --rm --gpus all -e NGC_API_KEY="$NGC_API_KEY" \
  --volume "$DATA":/workdir/dataset --volume "$OUT/aux":/workdir/output \
  nvcr.io/nvidia/nre/nre-tools-ga:$TAG \
  --dataset-path=/workdir/dataset/"$SEQ" --output-dir=/workdir/output \
  --store-meta --no-seg-logits --lidar-seg-camvis
cp -r "$OUT"/aux/*.aux.* "$DATA"/          # the aux stores must sit next to the manifest

# 2 training (static background + dynamic_rigids / dynamic_deformables layers from the GT_SYNTHETIC cuboids)
# The production config is written for NVIDIA's Hyperion 8.1 rig (f-theta cameras); our rig is 6 pinhole cameras
# (90 deg HFOV, 1280x720) and a 64-beam lidar - check the parsed config in the output if training complains.
docker run --shm-size=64g --rm --gpus all -e NGC_API_KEY="$NGC_API_KEY" \
  --volume "$DATA":/workdir/dataset --volume "$OUT/train":/workdir/output \
  nvcr.io/nvidia/nre/nre-ga:$TAG \
  mode=train out_dir=/workdir/output \
  --config-name=configs/apps/prod/Hyperion-8.1/car2sim_6cam.yaml \
  dataset.path=/workdir/dataset/"$SEQ" \
  "dataset.camera_ids=[${CAMS}]" "dataset.lidar_ids=[lidar_top]" dataset.aux_data=True \
  trainer.max_epochs="$EPOCHS"

echo "result: $(ls -d "$OUT"/train/*/usd-out/last.usdz 2>/dev/null || echo 'no usdz - see the logs above')"
