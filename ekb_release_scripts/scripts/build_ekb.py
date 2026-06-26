#!/usr/bin/env python3
"""Build an External Knowledge Bank (EKB) from training query images.

The pipeline:
1. Uses Google Cloud Vision Web Detection to retrieve visually similar images.
2. Downloads the top candidate URLs for each query.
3. Extracts ArcFace-compatible face embeddings with InsightFace.
4. Filters candidates by ArcFace distance to the query image.
5. Deduplicates images by SHA-256 hash and writes reproducibility manifests.

The script intentionally records detailed metadata so that an EKB can be
audited and reconstructed without redistributing web images.
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
import sys
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
    sha256: str
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
    retained_face_path: str
    filter_status: str
    filter_reason: str
    arcface_distance: float | str
    face_bbox: str
    image_sha256: str
    face_sha256: str


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
    image_sha256: str
    face_sha256: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Construct an External Knowledge Bank with web retrieval and ArcFace filtering."
    )
    parser.add_argument("--queries", required=True, type=Path, help="CSV with at least query_id,image_path columns.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory for images and manifests.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of visually similar URLs to request per query.")
    parser.add_argument("--min-arcface-dist", type=float, default=0.05, help="Lower ArcFace distance threshold.")
    parser.add_argument("--max-arcface-dist", type=float, default=0.73, help="Upper ArcFace distance threshold.")
    parser.add_argument("--max-retained-per-query", type=int, default=2, help="Maximum retained candidates per query.")
    parser.add_argument("--select-per-query", type=int, default=1, help="Number of final EKB samples selected per query.")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed for final candidate selection.")
    parser.add_argument("--timeout", type=float, default=12.0, help="HTTP download timeout in seconds.")
    parser.add_argument("--max-image-mb", type=float, default=12.0, help="Maximum candidate image size in MB.")
    parser.add_argument("--sleep", type=float, default=0.0, help="Optional sleep between Google Vision calls.")
    parser.add_argument("--resume", action="store_true", help="Reuse existing candidate downloads when present.")
    parser.add_argument("--skip-retrieval", action="store_true", help="Skip Google retrieval and use existing retrieved manifest.")
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

    # Fallback: some queries return matching images rather than visually similar images.
    for group_name in ("full_matching_images", "partial_matching_images"):
        for item in getattr(web_detection, group_name, []) or []:
            if item.url:
                urls.append(item.url)

    seen: set[str] = set()
    unique_urls: list[str] = []
    for url in urls:
        if url not in seen:
            unique_urls.append(url)
            seen.add(url)
        if len(unique_urls) >= top_k:
            break
    return unique_urls


def download_image(url: str, path: Path, timeout: float, max_image_mb: float) -> tuple[str, str, str, int, str]:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; EKBBuilder/1.0; "
            "+https://github.com/)"
        )
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
        image_hash = sha256_bytes(data)
        path.write_bytes(data)
        return "ok", "", content_type, len(data), image_hash
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
                sha256=str(row["sha256"]),
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
                image_hash = sha256_file(local_path)
            else:
                status, error, content_type, num_bytes, image_hash = download_image(
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
                    sha256=image_hash,
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
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return image
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
    embedding = embedding / norm
    return embedding, [x1, y1, x2, y2], crop


def arcface_distance(a: np.ndarray, b: np.ndarray) -> float:
    # Embeddings are L2-normalized, so Euclidean distance is stable and easy to audit.
    return float(np.linalg.norm(a - b))


def save_face_crop(crop_bgr: np.ndarray, out_path: Path) -> str:
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(crop_rgb)
    image = image.resize((224, 224), Image.Resampling.BICUBIC)
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
        path = Path(str(row["image_path"]))
        image = read_image_bgr(path)
        if image is None:
            LOGGER.warning("Cannot read query image: %s", path)
            continue
        result = largest_face(app, image)
        if result is None:
            LOGGER.warning("No face found in query image: %s", path)
            continue
        embedding, _, _ = result
        query_embeddings[query_id] = (embedding, str(row.get("label", "")))

    rows: list[FilteredCandidate] = []
    seen_hashes: set[str] = set()
    retained_counts: dict[str, int] = {}

    for item in tqdm(retrieved, desc="Filtering candidates"):
        candidate_path = Path(item.local_path)
        retained_face_path = ""
        filter_status = "rejected"
        filter_reason = ""
        distance_value: float | str = ""
        bbox_value = ""
        face_hash = ""

        if item.download_status != "ok" or not candidate_path.exists():
            filter_reason = "download_failed"
        elif item.sha256 in seen_hashes:
            filter_reason = "duplicate_image_hash"
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
                    candidate_embedding, bbox, crop = face_result
                    query_embedding, _ = query_embeddings[item.query_id]
                    dist = arcface_distance(query_embedding, candidate_embedding)
                    distance_value = round(dist, 6)
                    bbox_value = json.dumps(bbox)
                    if dist <= args.min_arcface_dist:
                        filter_reason = "near_duplicate_or_same_identity"
                    elif dist > args.max_arcface_dist:
                        filter_reason = "too_dissimilar"
                    elif retained_counts.get(item.query_id, 0) >= args.max_retained_per_query:
                        filter_reason = "query_retained_limit_reached"
                    else:
                        out_name = f"{safe_name(item.query_id)}_{safe_name(item.candidate_id)}.jpg"
                        out_path = dirs["filtered_faces"] / out_name
                        face_hash = save_face_crop(crop, out_path)
                        retained_face_path = str(out_path)
                        filter_status = "retained"
                        filter_reason = "passed"
                        retained_counts[item.query_id] = retained_counts.get(item.query_id, 0) + 1
                        seen_hashes.add(item.sha256)

        rows.append(
            FilteredCandidate(
                query_id=item.query_id,
                query_image_path=item.query_image_path,
                query_label=item.query_label,
                candidate_id=item.candidate_id,
                source_url=item.source_url,
                source_domain=item.source_domain,
                candidate_path=str(candidate_path),
                retained_face_path=retained_face_path,
                filter_status=filter_status,
                filter_reason=filter_reason,
                arcface_distance=distance_value,
                face_bbox=bbox_value,
                image_sha256=item.sha256,
                face_sha256=face_hash,
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
        if item.filter_status == "retained" and item.retained_face_path:
            retained_by_query.setdefault(item.query_id, []).append(item)

    rows: list[FinalPair] = []
    for query_id in sorted(retained_by_query):
        candidates = retained_by_query[query_id]
        candidates.sort(key=lambda item: float(item.arcface_distance))
        if len(candidates) > select_per_query:
            # Sample among retained candidates to avoid always choosing the closest face.
            selected = rng.sample(candidates, select_per_query)
            selected.sort(key=lambda item: float(item.arcface_distance))
        else:
            selected = candidates

        for item in selected:
            src = Path(item.retained_face_path)
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
                    image_sha256=item.image_sha256,
                    face_sha256=item.face_sha256,
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
    retained = sum(1 for row in filtered if row.filter_status == "retained")
    summary = {
        "num_queries": int(len(queries)),
        "num_retrieved_records": len(retrieved),
        "num_successful_downloads": retrieved_ok,
        "num_retained_faces": retained,
        "num_final_pairs": len(pairs),
        "download_success_rate": round(retrieved_ok / max(len(retrieved), 1), 6),
        "retention_rate_among_downloads": round(retained / max(retrieved_ok, 1), 6),
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
