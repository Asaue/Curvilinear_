# UCS RGB Eccentric Adapter Notes

## Goal

This document describes the latest UCS improvement: adding a lightweight RGB local eccentricity enhancement adapter before SAM/UCS image patch embedding.

The goal is to emphasize curve-like targets, cracks, scratches, and object boundaries where the center pixel differs strongly from its local neighborhood.

Core idea:

```text
center pixel differs a lot from neighborhood -> amplify
center pixel differs little from neighborhood -> mostly keep unchanged
```

This module is not intended to replace the original UCS ViT Adapter. Current experiments suggest it works best as an auxiliary input-side enhancement combined with the original UCS adapter.

## New Module

File:

```text
UCS/UCS/segment_anything/modeling/image_encoder.py
```

Add this class after `Adapter_Layer`:

```python
class RGBEccentricAdapter(nn.Module):
    def __init__(self, channels: int = 3, hidden_channels: int = 16, kernel_sizes=(3, 7, 15)):
        super().__init__()
        self.pads = nn.ModuleList([nn.ReflectionPad2d(kernel_size // 2) for kernel_size in kernel_sizes])
        self.pools = nn.ModuleList([nn.AvgPool2d(kernel_size, stride=1) for kernel_size in kernel_sizes])
        diff_channels = channels * len(kernel_sizes)

        self.fuse = nn.Conv2d(diff_channels, channels, kernel_size=1, bias=False)
        self.gate = nn.Sequential(
            nn.Conv2d(diff_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.alpha = nn.Parameter(torch.zeros(1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        diffs = []
        for pad, pool in zip(self.pads, self.pools):
            local_mean = pool(pad(x))
            diffs.append(x - local_mean)

        local_diff = torch.cat(diffs, dim=1)
        contrast_gate = self.gate(torch.abs(local_diff))
        eccentric_diff = self.fuse(local_diff)
        return x + self.alpha * eccentric_diff * contrast_gate
```

Formula:

```text
diff_k = x - AvgPool_k(x), k in {3, 7, 15}
gate = Sigmoid(Conv(ReLU(Conv(abs(concat(diff_k))))))
enhanced = x + alpha * Conv1x1(concat(diff_k)) * gate
```

`alpha` is initialized to `0`, so the adapter starts as nearly identity and learns how much local eccentric enhancement is useful during training.

## ImageEncoderViT Integration

File:

```text
UCS/UCS/segment_anything/modeling/image_encoder.py
```

Add a constructor argument:

```python
use_rgb_eccentric: bool = True
```

Inside `__init__`:

```python
self.use_rgb_eccentric = use_rgb_eccentric
self.RGBEccentricAdapter = RGBEccentricAdapter(in_chans)
```

In `forward`, replace the original input handling:

```python
inp = x
x = self.patch_embed(x)
```

with:

```python
inp = self.RGBEccentricAdapter(x) if self.use_rgb_eccentric else x
x = self.patch_embed(inp)
```

The existing prompt generator should keep using `inp`:

```python
handcrafted_feature = self.prompt_generator.init_handcrafted(inp)
```

So the RGB eccentric adapter affects both patch embedding and the handcrafted/prompt branch.

## build_sam.py Changes

File:

```text
UCS/UCS/segment_anything/build_sam.py
```

When calling `_build_sam` from `build_sam_vit_*`, pass:

```python
use_rgb_eccentric = getattr(args, "use_rgb_eccentric", True)
```

Add `_build_sam` argument:

```python
use_rgb_eccentric=True
```

Pass it into `ImageEncoderViT`:

```python
use_rgb_eccentric = use_rgb_eccentric
```

## train.py Changes

File:

```text
UCS/UCS/train.py
```

Fix boolean parsing first:

```python
def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ('true', '1', 'yes', 'y'):
        return True
    if value in ('false', '0', 'no', 'n'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')
```

Recommended arguments:

```python
parser.add_argument("--encoder_adapter", type=str2bool, default=True, help="use adapter")
parser.add_argument("--use_rgb_eccentric", type=str2bool, default=True, help="use RGB eccentric adapter before patch embedding")
parser.add_argument("--save_periodic", type=str2bool, default=True, help="save periodic checkpoints every 200 batches")
```

To save only the best checkpoint, change periodic saving:

```python
if args.save_periodic and int(batch+1) % 200 == 0:
```

Keep the original best-checkpoint logic.

## Important Batch Size Note

The current UCS `mask_decoder.py` PCA branch has a hidden batch-size assumption. Testing `batch_size=2` caused:

```text
RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x50 and 25x256)
```

Unless the PCA branch is fixed/vectorized, use:

```text
--batch_size 1
```

On a large-memory machine, prefer running multiple experiments in parallel, or first ask Codex to fix the PCA branch batch handling before increasing batch size.

## Training Commands

### New method: UCS Adapter + RGB Eccentric Adapter

```bash
cd /root/autodl-tmp/UCS/UCS

/root/miniconda3/bin/python train.py \
  --device cuda \
  --model_type vit_l \
  --image_size 1024 \
  --batch_size 1 \
  --epochs 500 \
  --iter_point 4 \
  --encoder_adapter true \
  --use_rgb_eccentric true \
  --save_periodic false \
  --data_path /root/autodl-tmp/1k_size \
  --sam_checkpoint /root/autodl-fs/workdir/models/sam_lds/sam_l_1024_v4_wavelet_pca_adapteronly_baseon1105.pth \
  --work_dir /root/autodl-tmp/UCS/UCS/weight \
  --run_name ucs_1k_rgb_eccentric_e500 \
  --save_path ucs_1k_rgb_eccentric_e500.pth \
  --save_path_bmt ucs_1k_rgb_eccentric_e500
```

### Baseline: original UCS Adapter, no RGB Eccentric Adapter

```bash
cd /root/autodl-tmp/UCS/UCS

/root/miniconda3/bin/python train.py \
  --device cuda \
  --model_type vit_l \
  --image_size 1024 \
  --batch_size 1 \
  --epochs 500 \
  --iter_point 4 \
  --encoder_adapter true \
  --use_rgb_eccentric false \
  --save_periodic false \
  --data_path /root/autodl-tmp/1k_size \
  --sam_checkpoint /root/autodl-fs/workdir/models/sam_lds/sam_l_1024_v4_wavelet_pca_adapteronly_baseon1105.pth \
  --work_dir /root/autodl-tmp/UCS/UCS/weight \
  --run_name ucs_1k_baseline_e500 \
  --save_path ucs_1k_baseline_e500.pth \
  --save_path_bmt ucs_1k_baseline_e500
```

## Dataset JSON

If `/root/autodl-tmp/1k_size/image2label_train.json` does not exist, generate it with:

```python
import json
from pathlib import Path

root = Path('/root/autodl-tmp/1k_size')
classes = ['branch', 'crack', 'floor', 'leaf', 'scratch', 'soil', 'tyre', 'wire']
exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
data = {}

for cls in classes:
    image_dir = root / cls
    label_dir = root / f'{cls}_label'
    for image in sorted(image_dir.iterdir()):
        if image.suffix.lower() not in exts:
            continue
        mask = label_dir / f'{image.stem}.png'
        if not mask.exists():
            candidates = sorted(label_dir.glob(image.stem + '.*'))
            if candidates:
                mask = candidates[0]
            else:
                continue
        data[str(image)] = [str(mask)]

json.dump(data, open(root / 'image2label_train.json', 'w'), indent=2)
print(len(data))
```

Current local count for `/root/autodl-tmp/1k_size`:

```text
original images: 850
normal labels: 850
centerline labels: 850
```

The generated training JSON uses the normal labels, not centerline labels.

## Observed Results So Far

### Sampling dataset

On `/root/autodl-tmp/sampling`, the new method improved early convergence strongly:

```text
UCS + RGB eccentric:
epoch 1 Dice 0.4168
epoch 6 Dice 0.4454

Old UCS baseline:
epoch 18 Dice 0.4313
```

### 1k_size dataset

On `/root/autodl-tmp/1k_size`, after 15 epochs:

```text
UCS + RGB eccentric best:
epoch 14
IoU 0.3635
Dice 0.4815

Old UCS baseline best:
epoch 13
IoU 0.3549
Dice 0.4730
```

The new method is slightly ahead so far, but the gap is not huge on 1k_size yet. Continue training and evaluate on a fixed validation/test split.

## Recommendations

1. Reproduce `use_rgb_eccentric=true` vs `use_rgb_eccentric=false` with all other settings identical.
2. Keep `--save_periodic false` if only best checkpoints are desired.
3. Do not increase batch size until the PCA branch in `mask_decoder.py` is fixed.
4. Do final comparison on a fixed validation/test set, not only training Dice.
