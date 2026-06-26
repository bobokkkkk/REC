# External Knowledge Bank Construction

This folder provides a reference implementation for constructing an External Knowledge Bank (EKB) from training images. The pipeline retrieves visually similar web images, filters candidate faces with ArcFace similarity constraints, removes near-duplicates, and writes reproducibility metadata.

## Requirements

Install the dependencies:

```bash
pip install -r requirements-ekb.txt
```

The retrieval stage uses Google Cloud Vision Web Detection. Set the service account credential before running:

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
```

On Windows PowerShell:

```powershell
$env:GOOGLE_APPLICATION_CREDENTIALS="C:\path\to\service-account.json"
```

## Input Query File

Prepare a CSV file containing training query images. The file must include:

```text
query_id,image_path
```

Optional columns such as `label`, `video_id`, and `split` are preserved in the output manifests.

Example:

```csv
query_id,image_path,label,video_id
train_000001,/data/ffpp/train/real/000001.png,real,000001
train_000002,/data/ffpp/train/fake/000002.png,fake,000002
```

Only training images should be used as queries. Test images or test labels should never be used during EKB construction.

## Run

```bash
python scripts/build_ekb.py \
  --queries data/train_queries.csv \
  --out-dir outputs/ekb \
  --top-k 5 \
  --min-arcface-dist 0.05 \
  --max-arcface-dist 0.73 \
  --max-retained-per-query 2 \
  --select-per-query 1
```

## Outputs

The script writes:

```text
outputs/ekb/
  candidates/                  # downloaded candidate images
  filtered_faces/              # cropped and retained face images
  selected_faces/              # final EKB images used for training pairs
  manifests/
    run_config.json
    retrieved_candidates.csv
    filtered_ekb_manifest.csv
    final_pairs.csv
```

The manifests include source URLs, source domains, download status, SHA-256 image hashes, ArcFace distances, filtering decisions, and selected training-pair indices.

## Notes

Web images may be subject to copyright, privacy, or platform-specific terms. When redistribution of images is not permitted, release the construction code and metadata rather than the images themselves.
