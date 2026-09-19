# REC

This repository contains the implementation of **REC (Reconstruction-Based
Extra-Data Contrastive Framework)** for cross-dataset deepfake detection.
REC uses visually related external face images together with
reconstruction-based feature learning and multi-level contrastive learning.

## Requirements

The code was developed with Python and PyTorch. The included
`requirements.txt` is a Windows Conda environment export. On a matching
platform, it can be used to create the environment with:

```bash
conda create --name rec --file requirements.txt
conda activate rec
```

If you use another platform or an existing environment, install the
corresponding PyTorch and Python packages manually.

The EKB construction script additionally uses Google Cloud Vision and
InsightFace. These packages are only needed if you want to build an EKB
yourself.

## Preparing the CLIP Model

The model loads the following local CLIP checkpoint:

```text
modules/clip-vit-base-patch32
```

Download the corresponding Hugging Face model files and place them in this
directory before training or testing. The directory should contain the usual
local files required by `CLIPModel.from_pretrained` and
`CLIPProcessor.from_pretrained`.

## Preparing the Datasets

Set the dataset paths in `configs/rec.yaml` before running the code:

```yaml
paths:
  extra_data_root: /path/to/extra_data
  ffpp: /path/to/FaceForensics++
  cdfv2: /path/to/Celeb-DF-v2
  dfdcp: /path/to/DFDC
  dfd: /path/to/Google-DFD
  wild: /path/to/Deepfake-in-the-Wild
```

The dataset key used on the command line must be one of:

```text
ffpp, cdfv2, dfdcp, dfd, wild
```

The expected file organization follows the dataset readers in `dataset/`.
Please check the corresponding reader if your local dataset has a different
directory layout.

## External Knowledge Bank

The EKB images used in the experiments are not included in this repository.
The corresponding EKB release package will be made available soon.

The repository includes a script for constructing an EKB from your own
training images:

```text
ekb_release_scripts/scripts/build_ekb.py
```

Only training images should be used as retrieval queries. Prepare a CSV file
with the following columns:

```text
query_id,image_path,label
```

Set Google Cloud Vision credentials before running the script:

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/google-credentials.json
```

Then run, for example:

```bash
python ekb_release_scripts/scripts/build_ekb.py \
  --queries /path/to/queries.csv \
  --out-dir /path/to/ekb_output \
  --device cuda
```

The script retrieves visually similar candidates, detects and crops faces,
removes exact and near duplicates, applies ArcFace-based filtering, and
writes the resulting manifests under the output directory.

## Training

After preparing the datasets, the local CLIP checkpoint, and the external
data path, run:

```bash
python main.py \
  --trainset ffpp \
  --valset cdfv2 \
  --testset cdfv2 \
  --config configs/rec.yaml \
  --epochs 30 \
  --save_path log/best_rec.pth
```

The same command is also available in `train.sh`.

## Evaluation

To evaluate a saved checkpoint:

```bash
python main.py \
  --test_only \
  --testset cdfv2 \
  --config configs/rec.yaml \
  --save_path log/best_rec.pth
```

The evaluation command is also provided in `test.sh`. Change `--testset` to
`ffpp`, `cdfv2`, `dfdcp`, `dfd`, or `wild` as needed.

## Repository Structure

```text
configs/                Configuration files
dataset/                Dataset readers
ekb_release_scripts/    EKB construction utilities
modules/                REC model components
main.py                 Training and evaluation entry point
train.sh                Example training command
test.sh                 Example evaluation command
```

## Notes

Make sure this module is
available in the repository or on `PYTHONPATH` before running the code.

## License

See `LICENSE` for the license of this repository.
