# Setup

## The short version

```bash
mamba env create -f environment.yml
mamba activate soccer-mocap
pip install -e .
python scripts/download_models.py mediapipe
```

Verify:

```bash
soccer-mocap backends     # mediapipe should say "ready"
pytest -q                 # 98 passing
```

## Why some packages come from pip

The environment is mamba-managed, and everything that has a clean
conda-forge package comes from conda-forge. Four entries deliberately do not,
and each one is there for a specific reason rather than convenience.

### PyTorch — pinned to the CUDA 12.8 wheels

```yaml
- --extra-index-url https://download.pytorch.org/whl/cu128
- torch==2.11.0+cu128
- torchvision==0.26.0+cu128
```

RTX 50-series cards are Blackwell (`sm_120`) and need CUDA 12.8. The
conda-forge builds are not compiled for that architecture.

The versions are pinned **with the `+cu128` local tag on purpose**. Because
this is an `--extra-index-url` rather than `--index-url`, PyPI stays in the
resolution set, and PyPI publishes higher `torch` versions that are CPU-only
on Windows. Left unpinned, pip picks the higher number and silently installs
a CPU build on a machine with a perfectly good GPU. That is exactly what
happened during the first build of this environment — `torch 2.14.0+cpu` on
an RTX 5070.

Check what you actually got:

```python
import torch
print(torch.__version__, torch.cuda.is_available())
print(torch.cuda.get_arch_list())      # want sm_120 for a 50-series card
```

Expected: `2.11.0+cu128 True` and `sm_120` in the list.

**No NVIDIA GPU?** Replace the two pinned lines with plain `torch` and
`torchvision` and drop the index URL. Nothing in the working pipeline needs
the GPU — MediaPipe is CPU-only. The CUDA stack is for the heavier backends.

### Pillow — from PyPI, not conda-forge

torchvision ships its own `jpeg8.dll`, `zlib.dll` and `libpng16.dll` inside
its package directory, and puts that directory on the Windows DLL search
path. conda-forge's Pillow then resolves `_imaging` against those
incompatible copies and dies:

```
ImportError: DLL load failed while importing _imaging:
The operating system cannot run %1.
```

The symptom is confusing because it is import-order dependent: `import PIL`
alone works, and `import torchvision` fails. The PyPI Pillow wheel bundles
its own codecs and is unaffected.

### OpenCV — `opencv-contrib-python`, not conda-forge `opencv`

mediapipe hard-depends on the pip distribution. Installing both puts two
different `cv2` modules on the path.

### mediapipe — no conda-forge build for Windows

Note also that **mediapipe 1.0 removed the `mp.solutions` API** that used to
bundle model weights inside the wheel. Only the Tasks API remains, and it
loads a `.task` file from disk, which is why the model download is a required
setup step:

```bash
python scripts/download_models.py mediapipe             # 'full', ~9 MB
python scripts/download_models.py mediapipe --variant heavy
python scripts/download_models.py --list
```

Models land in `models/`, which is gitignored.

## Verifying the whole stack

Import order matters for the Pillow/torchvision issue above, so test the
worst case — torchvision first:

```python
import torchvision, torch, cv2, PIL, mediapipe, gradio, av
print("all OK")
```

## Optional backends

None of these are needed for the working pipeline; each is a registered stub
with its integration notes in its own module.

```bash
# RTMPose — fast multi-person, foot keypoints
mamba run -n soccer-mocap pip install rtmlib onnxruntime-gpu

# Sapiens — highest accuracy, needs a GPU and a multi-GB checkpoint
python scripts/download_models.py sapiens      # prints instructions

# PosePipeline — full provenance tracking, needs MySQL
mamba run -n soccer-mocap pip install datajoint
```

`soccer-mocap backends` prints the exact command for whatever is missing.

## Troubleshooting

**`Could not open <file>` on an iPhone `.mov`** — the codec is HEVC. PyAV
handles it and is in the environment; if it is somehow missing:
`mamba install -n soccer-mocap -c conda-forge av`.

**Everyone is sideways** — the container rotation flag was not applied.
`soccer-mocap probe clip.mov` shows the flag. `video.apply_rotation` should
be `true` (the default).

**No people detected** — check the subject is large enough in frame, and that
`soccer-mocap probe` reports a sensible display size. Pose models fail almost
totally on a sideways person.

**`No pose backend is installed`** — you skipped
`python scripts/download_models.py mediapipe`.

**Velocities in the hundreds of body-heights per second** — this was a real
bug, fixed by estimating stature from summed segment lengths and using one
robust reference per track. If you see it again with a new backend, the
stature estimate is collapsing; check `body_height_px` on the frames
involved.

## Recreating the environment from scratch

```bash
mamba env remove -n soccer-mocap
mamba env create -f environment.yml
mamba activate soccer-mocap
pip install -e .
python scripts/download_models.py mediapipe
pytest -q
```
