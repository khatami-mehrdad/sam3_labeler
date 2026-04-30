# SAM 3 Labeler

Utilities for auto-labeling restaurant images and videos with Meta SAM 3 / SAM 3.1.

The goal of this project is to use SAM 3 as an offline labeling teacher:

- generate bounding boxes and masks for restaurant objects;
- propagate labels through video clips where SAM 3 tracking is available;
- export labels that can be reviewed and then used to train smaller runtime models.

SAM 3 is not vendored in this repository. Install it in a separate Python 3.12 CUDA environment from:

https://github.com/facebookresearch/sam3

## Setup

SAM 3 currently requires Python 3.12+, PyTorch 2.7+, and CUDA 12.6+.

```bash
python3.12 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip "setuptools<81" wheel
python -m pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128

mkdir -p external
git clone https://github.com/facebookresearch/sam3.git external/sam3
python -m pip install -e "external/sam3[notebooks]"
python -m pip install -e .
```

Request access to the SAM 3 checkpoints on Hugging Face, then authenticate:

```bash
hf auth login
```

## Image Labeling

```bash
python scripts/label_images.py \
  --input /path/to/images \
  --output /path/to/sam3_labels \
  --ontology configs/restaurant_ontology.yaml \
  --checkpoint checkpoints/sam3.1/sam3.1_multiplex.pt \
  --score-threshold 0.35 \
  --save-masks \
  --save-yolo
```

For large datasets, use `--input-root` so outputs mirror the input tree:

```bash
python scripts/label_images.py \
  --input /data1/ml_data/resturant_dataset/Cafe_Dataset/Cafe_Dataset/Dataset/cafe \
  --input-root /data1/ml_data/resturant_dataset/Cafe_Dataset/Cafe_Dataset/Dataset/cafe \
  --output outputs/cafe_full_single_gpu \
  --ontology configs/restaurant_ontology.yaml \
  --checkpoint checkpoints/sam3.1/sam3.1_multiplex.pt \
  --score-threshold 0.35 \
  --resume \
  --save-masks \
  --save-yolo
```

Per-image annotation JSON is the default metadata format. For an input image such as
`20/283/images/frames_192.jpg`, outputs are written as:

- `annotations/20/283/images/frames_192.json`
- `masks/20/283/images/frames_192.npz` when `--save-masks` is enabled
- `yolo/20/283/images/frames_192.txt` when `--save-yolo` is enabled

Add `--save-jsonl` only if you also want a top-level `annotations.jsonl` index.

NMS is disabled by default so raw SAM detections are preserved. If you want a cleaned
review export, add `--nms-iou-threshold 0.85` or another threshold explicitly.

For a faster full run, choose GPUs with enough free memory and launch one worker per
GPU. Each worker still labels one image at a time, but the workers share the same output
tree and skip completed per-image JSON files on resume:

```bash
python scripts/launch_multi_gpu_images.py \
  --gpus 0 2 4 6 7 \
  --input /data1/ml_data/resturant_dataset/Cafe_Dataset/Cafe_Dataset/Dataset/cafe \
  --input-root /data1/ml_data/resturant_dataset/Cafe_Dataset/Cafe_Dataset/Dataset/cafe \
  --output outputs/cafe_full_common_multigpu \
  --ontology configs/restaurant_common_ontology.yaml \
  --checkpoint checkpoints/sam3.1/sam3.1_multiplex.pt \
  --score-threshold 0.35 \
  --resume \
  --save-masks \
  --save-yolo \
  --detach
```

The current GPU state can be checked with:

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv
```

Outputs:

- `annotations/**/*.json`: one JSON record per image.
- `masks/**/*.npz`: compressed binary masks when `--save-masks` is enabled.
- `yolo/**/*.txt`: YOLO box labels when `--save-yolo` is enabled.
- `classes.txt`: class order used by YOLO exports.

Visualize a labeled image:

```bash
python scripts/visualize_image_labels.py \
  --annotations /path/to/sam3_labels/annotations \
  --input-root /path/to/images \
  --image /path/to/images/example.jpg \
  --output /path/to/sam3_labels/example_overlay.png \
  --score-threshold 0.35
```

## Video Labeling

```bash
python scripts/label_video.py \
  --video /path/to/clip.mp4 \
  --output /path/to/video_labels \
  --ontology configs/restaurant_ontology.yaml \
  --checkpoint checkpoints/sam3.1/sam3.1_multiplex.pt \
  --frame-index 0 \
  --gpus 4
```

`--video` may be either an MP4 file or a directory of consecutive JPEG frames. For frame directories, use lexicographically sortable names such as `000001.jpg`, `000002.jpg`, etc.

The video script uses SAM 3.1 by default through `build_sam3_predictor(version="sam3.1")`. It runs each ontology prompt in a separate SAM 3.1 session. This matters because the SAM 3 example notebook resets the session when changing text prompts. For a prompt such as `person`, SAM 3.1 detects all matching instances, assigns object IDs, and propagates their masks through the video. The class label is inherited from the prompt/class mapping in `restaurant_ontology.yaml`.

Outputs are stored per class:

- `prompt_frame_outputs.pt`: raw SAM 3 output on the prompted frame.
- `tracked_outputs_per_frame.pt`: raw propagated frame outputs when propagation is enabled.
- `prompt_frame_summary.json`: lightweight schema summary for inspection.
- `summary.json`: one record per prompt/class.

## Recommended Workflow

1. Sample frames from Insper/Cafe/restaurant footage.
2. Run `label_images.py` with the restaurant ontology.
3. Review a subset manually and tune prompts/thresholds.
4. Use SAM 3 tracking on short video clips to expand labels.
5. Train YOLO/segmentation/action models only after review.

Use SAM 3 for spatial labels. Use a video VLM such as Gemini 3.1 Pro for action/state weak labels like `ordering`, `eating`, `leaving`, and `bussing`.
