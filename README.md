# Curvilinear Segmentation with UCS + RGB Eccentric Adapter

This repository contains the code used for curvilinear structure segmentation experiments on cracks, wires, scratches, leaves, branches, floor edges, soil traces, and tyre marks. The model is based on UCS/SAM-style promptable segmentation and adds an **RGB Eccentric Adapter** before the ViT patch embedding to amplify local RGB differences around thin and boundary-like structures.

## What is included

- `train.py`: fine-tuning entry point.
- `test.py`: validation/testing entry point.
- `segment_anything/`: modified SAM/UCS model code, including the UCS adapters and the RGB Eccentric Adapter.
- `DataLoader.py`: dataset loader for image/label JSON mappings.
- `metrics.py`, `metric.py`, `utils.py`: loss, prompt generation, saving, and metric utilities.
- `tools/make_label2image_json.py`: converts training-style `image2label_train.json` into the `label2image_test.json` format expected by `test.py`.

Large datasets, checkpoints, logs, generated masks, and local environments are intentionally excluded from GitHub.

## Method

The RGB Eccentric Adapter is implemented in `segment_anything/modeling/image_encoder.py` as `RGBEccentricAdapter`. It works directly on the normalized RGB input before patch embedding:

1. Compute local RGB means at multiple spatial scales.
2. Estimate local contrast as the difference between the original RGB value and nearby context.
3. Use a learned gate to enlarge high local differences and suppress weak local differences.
4. Add the enhanced RGB residual back to the input before the standard SAM/UCS image encoder.

This is controlled by `--use_rgb_eccentric true/false`. The original UCS adapter path is still controlled by `--encoder_adapter`.

## Environment

The recommended environment lives under the repository `envs/` directory and uses Python `venv`:

```bash
python3 -m venv envs/curvilinear
source envs/curvilinear/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Install a PyTorch build that matches your CUDA driver if the default `pip install torch torchvision` does not select the right CUDA wheel for your machine.

## Dataset

The curated dataset is hosted on Hugging Face:

<https://huggingface.co/datasets/Asaue/Curvilinear_data>

Download it into `data/`:

```bash
huggingface-cli download Asaue/Curvilinear_data \
  --repo-type dataset \
  --local-dir data/Curvilinear_data
```

The main curated split is expected at:

```text
data/Curvilinear_data/1k_size_handle
```

It contains 792 image/label pairs across 8 categories: branch, crack, floor, leaf, scratch, soil, tyre, and wire.

For `test.py`, generate the validation mapping once:

```bash
python tools/make_label2image_json.py \
  --image2label data/Curvilinear_data/1k_size_handle/image2label_train.json \
  --out data/Curvilinear_data/1k_size_handle/label2image_test.json
```

## Checkpoints

Place checkpoints under `checkpoints/`. Typical files used in the experiments were:

```text
checkpoints/sam_l_1024_v4_wavelet_pca_adapteronly_baseon1105.pth
checkpoints/ucs_1k_rgb_eccentric_e500_best.pth
```

The first one is the UCS/SAM initialization checkpoint. The second one is the best RGB Eccentric Adapter fine-tuned checkpoint.

## Fine-tuning

RGB Eccentric Adapter + UCS fine-tuning:

```bash
python train.py \
  --data_path data/Curvilinear_data/1k_size_handle \
  --work_dir weight \
  --run_name ucs_rgb_eccentric_1k \
  --model_type vit_l \
  --image_size 1024 \
  --batch_size 4 \
  --epochs 500 \
  --lr 1e-4 \
  --sam_checkpoint checkpoints/sam_l_1024_v4_wavelet_pca_adapteronly_baseon1105.pth \
  --encoder_adapter true \
  --use_rgb_eccentric true \
  --save_periodic false \
  --save_path_bmt ucs_rgb_eccentric_1k
```

Original UCS baseline without the RGB Eccentric Adapter:

```bash
python train.py \
  --data_path data/Curvilinear_data/1k_size_handle \
  --work_dir weight \
  --run_name ucs_baseline_1k \
  --model_type vit_l \
  --image_size 1024 \
  --batch_size 4 \
  --epochs 500 \
  --lr 1e-4 \
  --sam_checkpoint checkpoints/sam_l_1024_v4_wavelet_pca_adapteronly_baseon1105.pth \
  --encoder_adapter true \
  --use_rgb_eccentric false \
  --save_periodic false \
  --save_path_bmt ucs_baseline_1k
```

For very large GPUs, increase `--batch_size` according to available memory.

## Validation

Validate the RGB Eccentric Adapter checkpoint:

```bash
python test.py \
  --data_path data/Curvilinear_data/1k_size_handle \
  --work_dir results \
  --run_name ucs_rgb_eccentric_1k \
  --model_type vit_l \
  --image_size 1024 \
  --sam_checkpoint checkpoints/ucs_1k_rgb_eccentric_e500_best.pth \
  --encoder_adapter true \
  --use_rgb_eccentric true \
  --iter_point 8 \
  --metrics iou dice f1 pre rec \
  --save_pred false
```

Validate the UCS baseline checkpoint by setting `--use_rgb_eccentric false` and passing the baseline checkpoint path.

## Reference Results

The following numbers were obtained with the UCS built-in `test.py` style validation. Small variation is expected because prompts are sampled stochastically.

| Dataset / Class | Checkpoint | IoU | Dice/F1 | Precision | Recall | Loss |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `1k_size_handle/scratch` | UCS baseline init | 38.47 | 53.82 | 61.18 | 54.77 | 0.7120 |
| `1k_size_handle/scratch` | UCS + RGB Eccentric | 49.56 | 64.83 | 59.86 | 73.83 | 0.5516 |
| `uniformv4_data_demo` | UCS + RGB Eccentric | 46.24 | 61.31 | 56.19 | 76.86 | 1.0189 |

## Code Notes

- New adapter: `segment_anything/modeling/image_encoder.py`, class `RGBEccentricAdapter`.
- Model construction: `segment_anything/build_sam.py`.
- Training flags: `--use_rgb_eccentric`, `--rgb_eccentric_only`, `--save_periodic`.
- Testing flags: `--use_rgb_eccentric`, `--save_pred`, `--seed`.

## Repository Hygiene

Do not commit datasets, model weights, logs, generated masks, or local virtual environments. They are ignored by `.gitignore` and should be distributed through Hugging Face or another artifact store.
