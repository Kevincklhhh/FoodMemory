# Phone-port report: kitchen-inventory pipeline

_Generated 2026-04-20 for SenSys '26 demo planning._

## Pipeline

`hands23 → DINOv3 → OWLv2`. Target device: **Pixel 9 Pro**.

## Repos to clone (on the laptop)

- `git clone https://github.com/pytorch/executorch.git` — runtime / Android AAR.
- `git clone https://github.com/ddshan/hand_object_detector.git` — hands23 replacement: a Faster-R-CNN-on-torchvision variant from the same author, free of detectron2, more transplantable to Android. Hands23 head weights still drop in.
- `git clone https://github.com/ddshan/hands23_detector.git` — reference for head definitions and `get_PF` source-of-truth.
- `git clone https://github.com/facebookresearch/dinov3.git` — DINOv3 model code.

## DINOv3 model choice

Swap the current **ViT-7B/16** (~27 GB) for the distilled **ViT-L/16** checkpoint: `facebook/dinov3-vitl16-pretrain-lvd1689m` (300 M params, ~600 MB fp16, embed dim 1024). Same family, distilled from the 7B teacher, drops in by changing `dinov3_vit7b16` → `dinov3_vitl16` in `03_dino_food_matching.py`.

## hands23 backbone

Use `ddshan/hand_object_detector` (torchvision Faster R-CNN) as the base instead of the detectron2-bound `hands23_detector`. Port the hands23 box predictor + `z_head` onto it.
