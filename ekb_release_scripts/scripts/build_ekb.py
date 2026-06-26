#!/usr/bin/env python3
"""Construct an External Knowledge Bank (EKB) from training query images.

Pipeline:
1. Retrieve visually similar candidate images with Google Cloud Vision Web Detection.
2. Download candidate images and record source metadata.
3. Detect and crop faces, then resize face crops to 224x224.
4. Apply image-level filtering on face crops using SHA-256 and perceptual hash.
5. Apply ArcFace-distance filtering between each query face and candidate face.
6. Select final EKB training pairs and write audit manifests.

Only training images should be used as queries. Test images or test labels must
not be used during EKB construction.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import random
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

import cv2
import numpy as np
import pandas as pd
import requests
from PIL import Image
from tqdm import tqdm


LOGGER = logging.getLogger("build_ekb")


@dataclass
class RetrievedCandidate:
    query_id: str
    query_image_path: str
    query_label: str
    rank: int
    source_url: str
    source_domain: str
    candidate_id: str
    local_path: str
    download_status: str
    download_error: str
    content_type: str
    num_bytes: int
    raw_image_sha256: str
    retrieval_time_utc: str


@dataclass
class FilteredCandidate:
    query_id: str
    query_image_path: str
    query_label: str
    candidate_id: str
    source_url: str
    source_domain: str
    candidate_path: str
    cropped_face_path: str
    filter_status: str
    filter_reason: str
    arcface_distance: float | str
    face_bbox: str
    raw_image_sha256: str
    face_sha256: str
    face_phash: str
    min_phash_hamming_to_retained: int | str


@dataclass
class FinalPair:
    query_id: str
    query_image_path: str
    query_label: str
    selected_candidate_id: str
    selected_face_path: str
    source_url: str
    source_domain: str
    arcface_distance: float
    raw_image_sha256: str
    face_sha256: str
    face_phash: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Construct an EKB with web retrieval, face-crop hash filtering, and ArcFace filtering."
    )
    parser.add_argument("--queries", required=True, type=Path, help="CSV with query_id,image_path columns.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of visually similar URLs per query.")
    parser.add_argument("--min-arcface-dist", type=float, default=0.05, help="Lower ArcFace distance threshold.")
    parser.add_argument("--max-arcface-dist", type=float, default=0.73, help="Upper ArcFace distance threshold.")
    parser.add_argument(
        "--phash-threshold",
        type=int,
        default=5,
        help="Reject a face crop if its pHash Hamming distance to a retained crop is <= this value.",
    )
    parser.add_argument("--max-retained-per-query", type=int, default=2, help="Maximum retained candidates per query.")
    parser.add_argument("--select-per-query", type=int, default=1, help="Final selected EKB faces per query.")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed for final selection.")
    parser.add_argument("--timeout", type=float, default=12.0, help="HTTP timeout in seconds.")
    parser.add_argument("--max-image-mb", type=float, default=12.0, help="Maximum download size in MB.")
    parser.add_argument("--sleep", type=float, default=0.0, help="Optional sleep between Vision API calls.")
    parser.add_argument("--resume", action="store_true", help="Reuse existing candidate downloads.")
    parser.add_argument(
        "--skip-retrieval",
        action="store_true",
        help="Skip Vision API retrieval and reuse manifests/retrieved_candidates.csv.",
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="InsightFace runtime device.")
    parser.add_argument("--det-size", type=int, default=640, help="InsightFace detector input size.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def ensure_dirs(out_dir: Path) -> dict[str, Path]:
    dirs = {
        "candidates": out_dir / "candidates",
        "face_crops": out_dir / "face_crops",
        "filtered_faces": out_dir / "filtered_faces",
        "selected_faces": out_dir / "selected_faces",
        "manifests": out_dir / "manifests",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def normalize_query_table(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path)
    required = {"query_id", "image_path"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"Missing required columns in query CSV: {sorted(missing)}")
    table["query_id"] = table["query_id"].astype(str)
    table["image_path"] = table["image_path"].astype(str)
    if "label" not in table.columns:
        table["label"] = ""
    table["label"] = table["label"].fillna("").astype(str)
    return table


def safe_name(text: str, max_len: int = 120) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")
    return text[:max_len] or "item"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def domain_from_url(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


def hamming_distance(hex_a: str, hex_b: str) -> int:
    if not hex_a or not hex_b:
        return 10**9
    return (int(hex_a, 16) ^ int(hex_b, 16)).bit_count()


def perceptual_hash_64(image: Image.Image) -> str:
    """Compute a 64-bit pHash represented as 16 hexadecimal characters."""
    gray = image.convert("L").resize((32, 32), Image.Resampling.LANCZOS)
    pixels = np.asarray(gray, dtype=np.float32)
    dct = cv2.dct(pixels)
    low_freq = dct[:8, :8].copy()
    values = low_freq.flatten()
    median = np.median(values[1:])  # Exclude DC component.
    bits = values > median
    value = 0
    for bit in bits:
        value = (value << 1) | int(bool(bit))
    return f"{value:016x}"


def load_google_client() -> Any:
    try:
        from google.cloud import vision
    except ImportError as exc:
        raise RuntimeError("google-cloud-vision is not installed.") from exc

    if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        raise RuntimeError("GOOGLE_APPLICATION_CREDENTIALS is not set.")
    return vision.ImageAnnotatorClient()


def retrieve_urls_google(client: Any, image_path: Path, top_k: int) -> list[str]:
    from google.cloud import vision

    with image_path.open("rb") as handle:
        image = vision.Image(content=handle.read())

    response = client.web_detection(image=image, max_results=top_k)
    if response.error.message:
        raise RuntimeError(response.error.message)

    web_detection = response.web_detection
    urls: list[str] = []
    for item in getattr(web_detection, "visually_similar_images", []) or []:
        if item.url:
            urls.append(item.url)

    # Fallback for queries where visually_similar_images is sparse.
    for group_name in ("full_matching_images", "partial_matching_images"):
        for item in getattr(web_detection, group_name, []) or []:
            if item.url:
                urls.append(item.url)

    unique_urls: list[str] = []
    seen: set[str] = set()
    for url in urls:
        if url in seen:
            continue
        unique_urls.append(url)
        seen.add(url)
        if len(unique_urls) >= top_k:
            break
    return unique_urls


def download_image(url: str, path: Path, timeout: float, max_image_mb: float) -> tuple[str, str, str, int, str]:
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; EKBBuilder/1.0; +https://github.com/)"
    }
    try:
        response = requests.get(url, headers=headers, timeout=timeout, stream=True)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        max_bytes = int(max_image_mb * 1024 * 1024)
        chunks: list[bytes] = []
        total = 0
        for chunk in response.iter_content(chunk_size=8192):
            if not chunk:
                continue
            total += len(chunk)
            if total > max_bytes:
                return "failed", "image_too_large", content_type, total, ""
            chunks.append(chunk)
        data = b"".join(chunks)
        if not data:
            return "failed", "empty_response", content_type, 0, ""
        raw_hash = sha256_bytes(data)
        path.write_bytes(data)
        return "ok", "", content_type, len(data), raw_hash
    except Exception as exc:
        return "failed", str(exc).replace("\n", " ")[:500], "", 0, ""


def write_csv(path: Path, rows: Iterable[Any], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            if hasattr(row, "__dataclass_fields__"):
                writer.writerow(asdict(row))
            else:
                writer.writerow(row)


def read_retrieved_manifest(path: Path) -> list[RetrievedCandidate]:
    table = pd.read_csv(path).fillna("")
    rows: list[RetrievedCandidate] = []
    for _, row in table.iterrows():
        raw_hash = str(row.get("raw_image_sha256", row.get("sha256", "")))
        rows.append(
            RetrievedCandidate(
                query_id=str(row["query_id"]),
                query_image_path=str(row["query_image_path"]),
                query_label=str(row.get("query_label", "")),
                rank=int(row["rank"]),
                source_url=str(row["source_url"]),
                source_domain=str(row["source_domain"]),
                candidate_id=str(row["candidate_id"]),
                local_path=str(row["local_path"]),
                download_status=str(row["download_status"]),
                download_error=str(row["download_error"]),
                content_type=str(row["content_type"]),
                num_bytes=int(row["num_bytes"]) if str(row["num_bytes"]) else 0,
                raw_image_sha256=raw_hash,
                retrieval_time_utc=str(row["retrieval_time_utc"]),
            )
        )
    return rows


def retrieve_and_download(args: argparse.Namespace, queries: pd.DataFrame, dirs: dict[str, Path]) -> list[RetrievedCandidate]:
    manifest_path = dirs["manifests"] / "retrieved_candidates.csv"
    if args.skip_retrieval:
        if not manifest_path.exists():
            raise FileNotFoundError(f"--skip-retrieval was set but {manifest_path} does not exist.")
        LOGGER.info("Loading existing retrieval manifest: %s", manifest_path)
        return read_retrieved_manifest(manifest_path)

    client = load_google_client()
    rows: list[RetrievedCandidate] = []
    for _, query in tqdm(queries.iterrows(), total=len(queries), desc="Retrieving"):
        query_id = str(query["query_id"])
        query_image = Path(str(query["image_path"]))
        query_label = str(query.get("label", ""))
        try:
            urls = retrieve_urls_google(client, query_image, args.top_k)
        except Exception as exc:
            LOGGER.warning("Retrieval failed for %s: %s", query_id, exc)
            urls = []

        for rank, url in enumerate(urls, start=1):
            candidate_id = f"{safe_name(query_id)}_r{rank:02d}"
            suffix = Path(urlparse(url).path).suffix.lower()
            if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
                suffix = ".jpg"
            local_path = dirs["candidates"] / f"{candidate_id}{suffix}"
            if args.resume and local_path.exists():
                status = "ok"
                error = ""
                content_type = ""
                num_bytes = local_path.stat().st_size
                raw_hash = sha256_file(local_path)
            else:
                status, error, content_type, num_bytes, raw_hash = download_image(
                    url=url,
                    path=local_path,
                    timeout=args.timeout,
                    max_image_mb=args.max_image_mb,
                )

            rows.append(
                RetrievedCandidate(
                    query_id=query_id,
                    query_image_path=str(query_image),
                    query_label=query_label,
                    rank=rank,
                    source_url=url,
                    source_domain=domain_from_url(url),
                    candidate_id=candidate_id,
                    local_path=str(local_path),
                    download_status=status,
                    download_error=error,
                    content_type=content_type,
                    num_bytes=num_bytes,
                    raw_image_sha256=raw_hash,
                    retrieval_time_utc=utc_now(),
                )
            )

        if args.sleep > 0:
            time.sleep(args.sleep)

    write_csv(manifest_path, rows, list(RetrievedCandidate.__dataclass_fields__.keys()))
    LOGGER.info("Wrote retrieval manifest: %s", manifest_path)
    return rows


def load_face_app(device: str, det_size: int) -> Any:
    try:
        from insightface.app import FaceAnalysis
    except ImportError as exc:
        raise RuntimeError("insightface is not installed.") from exc

    providers = ["CPUExecutionProvider"]
    ctx_id = -1
    if device == "cuda":
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        ctx_id = 0

    app = FaceAnalysis(name="buffalo_l", providers=providers)
    app.prepare(ctx_id=ctx_id, det_size=(det_size, det_size))
    return app


def read_image_bgr(path: Path) -> np.ndarray | None:
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def largest_face(app: Any, image_bgr: np.ndarray) -> tuple[np.ndarray, list[int], np.ndarray] | None:
    faces = app.get(image_bgr)
    if not faces:
        return None
    face = max(faces, key=lambda item: max(0.0, (item.bbox[2] - item.bbox[0]) * (item.bbox[3] - item.bbox[1])))
    bbox = [int(round(x)) for x in face.bbox.tolist()]
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = image_bgr[y1:y2, x1:x2].copy()
    embedding = np.asarray(face.normed_embedding, dtype=np.float32)
    norm = np.linalg.norm(embedding)
    if norm == 0:
        return None
    return embedding / norm, [x1, y1, x2, y2], crop


def arcface_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def face_crop_to_pil(crop_bgr: np.ndarray) -> Image.Image:
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(crop_rgb)
    return image.resize((224, 224), Image.Resampling.BICUBIC)


def save_pil_jpeg(image: Image.Image, out_path: Path) -> str:
    image.save(out_path, quality=95)
    return sha256_file(out_path)


def filter_candidates(
    args: argparse.Namespace,
    queries: pd.DataFrame,
    retrieved: list[RetrievedCandidate],
    dirs: dict[str, Path],
) -> list[FilteredCandidate]:
    app = load_face_app(args.device, args.det_size)
    query_by_id = {str(row["query_id"]): row for _, row in queries.iterrows()}
    query_embeddings: dict[str, tuple[np.ndarray, str]] = {}

    for query_id, row in tqdm(query_by_id.items(), desc="Encoding queries"):
        query_path = Path(str(row["image_path"]))
        image = read_image_bgr(query_path)
        if image is None:
            LOGGER.warning("Cannot read query image: %s", query_path)
            continue
        result = largest_face(app, image)
        if result is None:
            LOGGER.warning("No face found in query image: %s", query_path)
            continue
        embedding, _, _ = result
        query_embeddings[query_id] = (embedding, str(row.get("label", "")))

    rows: list[FilteredCandidate] = []
    seen_face_sha256: set[str] = set()
    retained_phashes: list[str] = []
    retained_counts: dict[str, int] = {}

    for item in tqdm(retrieved, desc="Filtering candidates"):
        candidate_path = Path(item.local_path)
        cropped_face_path = ""
        filter_status = "rejected"
        filter_reason = ""
        distance_value: float | str = ""
        bbox_value = ""
        face_hash = ""
        face_phash = ""
        min_phash_distance: int | str = ""

        if item.download_status != "ok" or not candidate_path.exists():
            filter_reason = "download_failed"
        elif item.query_id not in query_embeddings:
            filter_reason = "missing_query_embedding"
        else:
            image = read_image_bgr(candidate_path)
            if image is None:
                filter_reason = "unreadable_image"
            else:
                face_result = largest_face(app, image)
                if face_result is None:
                    filter_reason = "no_detected_face"
                else:
                    candidate_embedding, bbox, crop_bgr = face_result
                    face_image = face_crop_to_pil(crop_bgr)
                    face_phash = perceptual_hash_64(face_image)
                    bbox_value = json.dumps(bbox)

                    crop_name = f"{safe_name(item.query_id)}_{safe_name(item.candidate_id)}.jpg"
                    crop_path = dirs["face_crops"] / crop_name
                    face_hash = save_pil_jpeg(face_image, crop_path)
                    cropped_face_path = str(crop_path)

                    if face_hash in seen_face_sha256:
                        filter_reason = "duplicate_face_sha256"
                    else:
                        if retained_phashes:
                            min_phash_distance = min(hamming_distance(face_phash, old) for old in retained_phashes)
                        else:
                            min_phash_distance = ""

                        if min_phash_distance != "" and int(min_phash_distance) <= args.phash_threshold:
                            filter_reason = "near_duplicate_face_phash"
                        else:
                            query_embedding, _ = query_embeddings[item.query_id]
                            dist = arcface_distance(query_embedding, candidate_embedding)
                            distance_value = round(dist, 6)
                            if dist <= args.min_arcface_dist:
                                filter_reason = "near_identical_face_arcface"
                            elif dist > args.max_arcface_dist:
                                filter_reason = "too_dissimilar_arcface"
                            elif retained_counts.get(item.query_id, 0) >= args.max_retained_per_query:
                                filter_reason = "query_retained_limit_reached"
                            else:
                                out_path = dirs["filtered_faces"] / crop_name
                                out_path.write_bytes(crop_path.read_bytes())
                                cropped_face_path = str(out_path)
                                filter_status = "retained"
                                filter_reason = "passed"
                                retained_counts[item.query_id] = retained_counts.get(item.query_id, 0) + 1
                                seen_face_sha256.add(face_hash)
                                retained_phashes.append(face_phash)

        rows.append(
            FilteredCandidate(
                query_id=item.query_id,
                query_image_path=item.query_image_path,
                query_label=item.query_label,
                candidate_id=item.candidate_id,
                source_url=item.source_url,
                source_domain=item.source_domain,
                candidate_path=str(candidate_path),
                cropped_face_path=cropped_face_path,
                filter_status=filter_status,
                filter_reason=filter_reason,
                arcface_distance=distance_value,
                face_bbox=bbox_value,
                raw_image_sha256=item.raw_image_sha256,
                face_sha256=face_hash,
                face_phash=face_phash,
                min_phash_hamming_to_retained=min_phash_distance,
            )
        )

    manifest_path = dirs["manifests"] / "filtered_ekb_manifest.csv"
    write_csv(manifest_path, rows, list(FilteredCandidate.__dataclass_fields__.keys()))
    LOGGER.info("Wrote filtered manifest: %s", manifest_path)
    return rows


def select_final_pairs(
    filtered: list[FilteredCandidate],
    dirs: dict[str, Path],
    select_per_query: int,
    seed: int,
) -> list[FinalPair]:
    rng = random.Random(seed)
    retained_by_query: dict[str, list[FilteredCandidate]] = {}
    for item in filtered:
        if item.filter_status == "retained" and item.cropped_face_path:
            retained_by_query.setdefault(item.query_id, []).append(item)

    rows: list[FinalPair] = []
    for query_id in sorted(retained_by_query):
        candidates = retained_by_query[query_id]
        candidates.sort(key=lambda item: float(item.arcface_distance))
        if len(candidates) > select_per_query:
            selected = rng.sample(candidates, select_per_query)
            selected.sort(key=lambda item: float(item.arcface_distance))
        else:
            selected = candidates

        for item in selected:
            src = Path(item.cropped_face_path)
            dst = dirs["selected_faces"] / src.name
            if not dst.exists():
                dst.write_bytes(src.read_bytes())
            rows.append(
                FinalPair(
                    query_id=item.query_id,
                    query_image_path=item.query_image_path,
                    query_label=item.query_label,
                    selected_candidate_id=item.candidate_id,
                    selected_face_path=str(dst),
                    source_url=item.source_url,
                    source_domain=item.source_domain,
                    arcface_distance=float(item.arcface_distance),
                    raw_image_sha256=item.raw_image_sha256,
                    face_sha256=item.face_sha256,
                    face_phash=item.face_phash,
                )
            )

    manifest_path = dirs["manifests"] / "final_pairs.csv"
    write_csv(manifest_path, rows, list(FinalPair.__dataclass_fields__.keys()))
    LOGGER.info("Wrote final pair manifest: %s", manifest_path)
    return rows


def write_run_config(args: argparse.Namespace, dirs: dict[str, Path], queries: pd.DataFrame) -> None:
    config = {
        "created_utc": utc_now(),
        "queries": str(args.queries),
        "num_queries": int(len(queries)),
        "top_k": args.top_k,
        "min_arcface_dist": args.min_arcface_dist,
        "max_arcface_dist": args.max_arcface_dist,
        "phash_threshold": args.phash_threshold,
        "max_retained_per_query": args.max_retained_per_query,
        "select_per_query": args.select_per_query,
        "seed": args.seed,
        "device": args.device,
        "det_size": args.det_size,
        "skip_retrieval": args.skip_retrieval,
    }
    path = dirs["manifests"] / "run_config.json"
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    LOGGER.info("Wrote run config: %s", path)


def summarize(
    queries: pd.DataFrame,
    retrieved: list[RetrievedCandidate],
    filtered: list[FilteredCandidate],
    pairs: list[FinalPair],
    dirs: dict[str, Path],
) -> None:
    retrieved_ok = sum(1 for row in retrieved if row.download_status == "ok")
    detected_faces = sum(1 for row in filtered if row.face_sha256)
    exact_duplicates = sum(1 for row in filtered if row.filter_reason == "duplicate_face_sha256")
    phash_duplicates = sum(1 for row in filtered if row.filter_reason == "near_duplicate_face_phash")
    retained = sum(1 for row in filtered if row.filter_status == "retained")
    summary = {
        "num_queries": int(len(queries)),
        "num_retrieved_records": len(retrieved),
        "num_successful_downloads": retrieved_ok,
        "num_detected_face_crops": detected_faces,
        "num_exact_duplicate_face_crops": exact_duplicates,
        "num_phash_near_duplicate_face_crops": phash_duplicates,
        "num_retained_faces": retained,
        "num_final_pairs": len(pairs),
        "download_success_rate": round(retrieved_ok / max(len(retrieved), 1), 6),
        "face_detection_rate_among_downloads": round(detected_faces / max(retrieved_ok, 1), 6),
        "retention_rate_among_detected_faces": round(retained / max(detected_faces, 1), 6),
    }
    path = dirs["manifests"] / "summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    LOGGER.info("Summary: %s", json.dumps(summary, indent=2))
    LOGGER.info("Wrote summary: %s", path)


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)
    random.seed(args.seed)
    np.random.seed(args.seed)

    dirs = ensure_dirs(args.out_dir)
    queries = normalize_query_table(args.queries)
    write_run_config(args, dirs, queries)

    retrieved = retrieve_and_download(args, queries, dirs)
    filtered = filter_candidates(args, queries, retrieved, dirs)
    pairs = select_final_pairs(filtered, dirs, args.select_per_query, args.seed)
    summarize(queries, retrieved, filtered, pairs, dirs)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        LOGGER.error("Interrupted.")
        raise SystemExit(130)
    except Exception as exc:
        LOGGER.exception("Failed: %s", exc)
        raise SystemExit(1)
