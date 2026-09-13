from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import joblib
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "dataset_3"
EXAMPLES_PATH = ROOT / "hackathon-intent-classifier-starter" / "examples.jsonl"
OUT_DIR = Path(__file__).resolve().parent
CACHE_DIR = OUT_DIR / "cache"
DATA_DIR = OUT_DIR / "data"
MODEL = os.getenv("GIGAEVO_EMBEDDING_MODEL", "EmbeddingsGigaR")
BATCH_SIZE = int(os.getenv("GIGAEVO_EMBEDDING_BATCH_SIZE", "32"))

LABELS = [
    "greeting",
    "capabilities",
    "gratitude",
    "data_catalog",
    "pivot_table",
    "technical",
    "support",
    "no_rag",
]
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
SPLIT_PATHS = {
    "train": DATASET_DIR / "train.jsonl",
    "validation": DATASET_DIR / "validation.jsonl",
    "test": EXAMPLES_PATH,
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def format_input(row: dict[str, Any]) -> str:
    history = row.get("history") or []
    history_lines = [
        f"{message.get('role', 'unknown')}: {message.get('content', '')}"
        for message in history
    ]
    history_text = "\n".join(history_lines) if history_lines else "<empty>"
    return f"HISTORY:\n{history_text}\nCURRENT_USER:\n{row.get('text', '')}"


def text_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def registry() -> tuple[str, str]:
    direct_key = os.getenv("GIGACHAT_API_KEY")
    if direct_key:
        return "https://api.giga.chat/v1", direct_key.strip()

    configured_path = os.getenv("GIGAEVO_LLM_REGISTRY")
    if not configured_path:
        raise FileNotFoundError(
            "GigaChat credentials are not configured. Set GIGAEVO_LLM_REGISTRY "
            "to GigaEVO's llm_models.yml, or set GIGACHAT_API_KEY."
        )
    path = Path(configured_path)
    if not path.exists():
        raise FileNotFoundError(f"GigaEVO LLM registry not found: {path}")
    config = path.read_text(encoding="utf-8")
    block_match = re.search(r"(?ms)- id: gigachat-max-2(.*?)(?=\n\s*- id:|\Z)", config)
    if not block_match:
        raise ValueError("gigachat-max-2 is absent from the GigaEVO LLM registry")
    block = block_match.group(1)
    base_match = re.search(r"(?m)^\s*base_url:\s*[\"']?([^\"'\r\n]+)", block)
    key_match = re.search(r"(?m)^\s*api_key:\s*[\"']?([^\"'\r\n]+)", block)
    if not base_match or not key_match:
        raise ValueError("gigachat-max-2 must define base_url and api_key")
    return base_match.group(1).strip().rstrip("/"), key_match.group(1).strip()


class GigaChatEmbeddings:
    def __init__(self) -> None:
        self.base_url, self.auth_key = registry()
        self.session = requests.Session()
        self.access_token = ""
        self.expires_at = 0.0

    def token(self) -> str:
        if self.access_token and time.time() < self.expires_at - 60:
            return self.access_token
        response = self.session.post(
            "https://ngw.devices.sberbank.ru:9443/api/v2/oauth",
            headers={
                "Authorization": f"Basic {self.auth_key}",
                "RqUID": str(uuid.uuid4()),
                "Accept": "application/json",
            },
            data={"scope": "GIGACHAT_API_PERS"},
            timeout=60,
            verify=False,
        )
        response.raise_for_status()
        payload = response.json()
        self.access_token = str(payload["access_token"])
        raw_expiry = float(payload.get("expires_at", 0))
        self.expires_at = raw_expiry / 1000 if raw_expiry > 10_000_000_000 else raw_expiry
        return self.access_token

    def embed(self, texts: list[str]) -> np.ndarray:
        """Request embeddings, retrying only errors which can resolve by themselves."""
        last_error: Exception | None = None
        for attempt in range(7):
            try:
                response = self.session.post(
                    "https://api.giga.chat/v1/embeddings",
                    headers={
                        "Authorization": f"Bearer {self.token()}",
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                    },
                    json={"model": MODEL, "input": texts},
                    timeout=180,
                    verify=False,
                )
                # One fresh token can resolve an expired access token. Repeating
                # a bad key indefinitely cannot, so fail on the next 401.
                if response.status_code == 401 and attempt == 0:
                    self.access_token = ""
                    continue

                # These responses describe a request, access or billing issue.
                # Retrying them with the same input only stalls the Streamlit
                # session for several minutes and then produces the same error.
                if response.status_code in {400, 401, 402, 403, 404, 413, 422}:
                    detail = response.text.strip().replace("\n", " ")[:500]
                    raise RuntimeError(
                        f"GigaChat embeddings request was rejected "
                        f"({response.status_code}): {detail or response.reason}"
                    )
                response.raise_for_status()
                data = sorted(response.json()["data"], key=lambda item: int(item["index"]))
                vectors = np.asarray([item["embedding"] for item in data], dtype=np.float32)
                if vectors.shape[0] != len(texts):
                    raise ValueError(
                        f"Embedding count mismatch: {vectors.shape[0]} != {len(texts)}"
                    )
                return vectors
            except RuntimeError:
                raise
            except Exception as exc:
                last_error = exc
                if attempt < 6:
                    time.sleep(min(45, 2**attempt))
        raise RuntimeError(f"Embedding request failed after retries: {last_error}")


def load_cache(path: Path) -> tuple[dict[str, np.ndarray], int]:
    if not path.exists():
        return {}, 0
    loaded = np.load(path, allow_pickle=False)
    keys = [str(value) for value in loaded["keys"]]
    matrix = np.asarray(loaded["embeddings"], dtype=np.float32)
    if len(keys) != matrix.shape[0]:
        raise ValueError(f"Corrupt embedding cache: {path}")
    return {key: matrix[index] for index, key in enumerate(keys)}, int(matrix.shape[1])


def save_cache(vectors: dict[str, np.ndarray]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    keys = list(vectors)
    matrix = np.asarray([vectors[key] for key in keys], dtype=np.float32)
    target = CACHE_DIR / "embeddings.npz"
    temporary = CACHE_DIR / "embeddings.tmp"
    with temporary.open("wb") as output:
        np.savez_compressed(output, keys=np.asarray(keys), embeddings=matrix)
    temporary.replace(target)


def build_permuted_rows(
    train_rows: list[dict[str, Any]], validation_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Arrange rows so GigaEVO's deterministic split equals dataset_3 validation."""
    total = len(train_rows) + len(validation_rows)
    test_size = len(validation_rows) / total
    per_label_train = {
        label: [row for row in train_rows if row["intent"] == label] for label in LABELS
    }
    per_label_validation = {
        label: [row for row in validation_rows if row["intent"] == label]
        for label in LABELS
    }
    slot_labels = np.concatenate(
        [
            np.full(
                len(per_label_train[label]) + len(per_label_validation[label]),
                LABEL_TO_ID[label],
                dtype=np.int64,
            )
            for label in LABELS
        ]
    )
    all_indices = np.arange(total)
    train_indices, validation_indices = train_test_split(
        all_indices,
        test_size=test_size,
        random_state=42,
        stratify=slot_labels,
    )
    train_index_set = set(int(index) for index in train_indices)
    validation_index_set = set(int(index) for index in validation_indices)
    arranged: list[dict[str, Any] | None] = [None] * total
    for label in LABELS:
        label_code = LABEL_TO_ID[label]
        label_train_slots = [
            index
            for index in range(total)
            if slot_labels[index] == label_code and index in train_index_set
        ]
        label_validation_slots = [
            index
            for index in range(total)
            if slot_labels[index] == label_code and index in validation_index_set
        ]
        if len(label_train_slots) != len(per_label_train[label]):
            raise AssertionError(f"Train slot mismatch for {label}")
        if len(label_validation_slots) != len(per_label_validation[label]):
            raise AssertionError(f"Validation slot mismatch for {label}")
        for index, row in zip(label_train_slots, per_label_train[label], strict=True):
            arranged[index] = row
        for index, row in zip(
            label_validation_slots, per_label_validation[label], strict=True
        ):
            arranged[index] = row
    if any(row is None for row in arranged):
        raise AssertionError("Not all GigaEVO row slots were populated")
    concrete = [row for row in arranged if row is not None]
    selected_ids = {concrete[int(index)]["id"] for index in validation_indices}
    expected_ids = {row["id"] for row in validation_rows}
    if selected_ids != expected_ids:
        raise AssertionError("GigaEVO's internal test split does not match validation")
    return concrete


def write_metadata(split: str, rows: list[dict[str, Any]]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / f"metadata_{split}.jsonl"
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            record = {
                "id": row["id"],
                "group_id": row.get("group_id"),
                "intent": row["intent"],
                "intent_code": LABEL_TO_ID[row["intent"]],
                "needs_clarification": bool(row.get("needs_clarification", False)),
                "tags": row.get("tags", []),
                "embedding_key": text_key(format_input(row)),
            }
            output.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_gigaevo_csv(
    arranged_rows: list[dict[str, Any]], vectors: dict[str, np.ndarray], dimension: int
) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / "gigaevo_train_validation_permuted.csv"
    columns = [f"f_{index:04d}" for index in range(dimension)]
    if path.exists():
        path.unlink()
    for start in range(0, len(arranged_rows), 256):
        batch = arranged_rows[start : start + 256]
        matrix = np.asarray(
            [vectors[text_key(format_input(row))] for row in batch], dtype=np.float32
        )
        frame = pd.DataFrame(matrix, columns=columns)
        frame["target"] = [LABEL_TO_ID[row["intent"]] for row in batch]
        frame.to_csv(
            path,
            mode="w" if start == 0 else "a",
            header=start == 0,
            index=False,
            float_format="%.9g",
        )
    return path


def build_projection(
    rows_by_split: dict[str, list[dict[str, Any]]],
    vectors: dict[str, np.ndarray],
    dimension: int,
) -> tuple[dict[str, np.ndarray], int]:
    """Project frozen embeddings to a compact representation for runner IPC."""
    projection_dimension = min(1024, dimension)
    train_matrix = np.asarray(
        [vectors[text_key(format_input(row))] for row in rows_by_split["train"]],
        dtype=np.float32,
    )
    pca = PCA(
        n_components=projection_dimension,
        svd_solver="randomized",
        random_state=42,
    )
    pca.fit(train_matrix)
    projected: dict[str, np.ndarray] = {}
    keys = list(vectors)
    for start in range(0, len(keys), 512):
        batch_keys = keys[start : start + 512]
        batch = np.asarray([vectors[key] for key in batch_keys], dtype=np.float32)
        transformed = np.asarray(pca.transform(batch), dtype=np.float32)
        projected.update({key: transformed[index] for index, key in enumerate(batch_keys)})
    projection_path = OUT_DIR / "gigaevo_pca.joblib"
    joblib.dump(pca, projection_path)
    np.savez_compressed(
        OUT_DIR / "cache" / "gigaevo_embeddings.npz",
        keys=np.asarray(keys),
        embeddings=np.asarray([projected[key] for key in keys], dtype=np.float32),
    )
    return projected, projection_dimension


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-csv", action="store_true")
    args = parser.parse_args()
    requests.packages.urllib3.disable_warnings()  # type: ignore[attr-defined]

    rows_by_split = {split: load_jsonl(path) for split, path in SPLIT_PATHS.items()}
    unique_texts: dict[str, str] = {}
    for rows in rows_by_split.values():
        for row in rows:
            formatted = format_input(row)
            key = text_key(formatted)
            if key in unique_texts and unique_texts[key] != formatted:
                raise ValueError(f"SHA-256 collision: {key}")
            unique_texts[key] = formatted

    current_path = CACHE_DIR / "embeddings.npz"
    vectors, dimension = load_cache(current_path)
    old_path = ROOT / "gigaevo_embedding_experiment_v1" / "cache" / "embeddings.npz"
    if not vectors and old_path.exists():
        old_vectors, old_dimension = load_cache(old_path)
        vectors = {key: old_vectors[key] for key in unique_texts if key in old_vectors}
        dimension = old_dimension if vectors else 0
        if vectors:
            save_cache(vectors)
            print(f"seeded_cache={len(vectors)} source={old_path}")

    vectors = {key: vector for key, vector in vectors.items() if key in unique_texts}
    missing = [(key, text) for key, text in unique_texts.items() if key not in vectors]
    print(
        f"records={sum(map(len, rows_by_split.values()))} "
        f"unique_inputs={len(unique_texts)} cached={len(vectors)} missing={len(missing)}"
    )
    if missing:
        client = GigaChatEmbeddings()
        for start in range(0, len(missing), BATCH_SIZE):
            batch = missing[start : start + BATCH_SIZE]
            batch_vectors = client.embed([text for _, text in batch])
            if dimension and batch_vectors.shape[1] != dimension:
                raise ValueError(
                    f"Embedding dimension changed: {batch_vectors.shape[1]} != {dimension}"
                )
            dimension = int(batch_vectors.shape[1])
            for (key, _), vector in zip(batch, batch_vectors, strict=True):
                vectors[key] = vector
            completed = start + len(batch)
            if (start // BATCH_SIZE + 1) % 8 == 0 or completed == len(missing):
                save_cache(vectors)
                print(f"embedded={completed}/{len(missing)} dimension={dimension}")

    if not dimension or len(vectors) != len(unique_texts):
        raise ValueError("Embedding cache is incomplete")
    save_cache(vectors)
    for split, rows in rows_by_split.items():
        write_metadata(split, rows)

    projected_vectors, projected_dimension = build_projection(
        rows_by_split, vectors, dimension
    )
    arranged_rows = build_permuted_rows(
        rows_by_split["train"], rows_by_split["validation"]
    )
    csv_path = DATA_DIR / "gigaevo_train_validation_permuted.csv"
    if args.force_csv or not csv_path.exists():
        csv_path = write_gigaevo_csv(
            arranged_rows, projected_vectors, projected_dimension
        )

    manifest = {
        "version": "gigaevo-embedding-experiment-dataset3-v1",
        "embedding_model": MODEL,
        "embedding_dimension": dimension,
        "gigaevo_feature_dimension": projected_dimension,
        "gigaevo_projection": "PCA(randomized, random_state=42), fit on dataset_3/train only",
        "input_format": "HISTORY role/content lines followed by CURRENT_USER text",
        "rows": {split: len(rows) for split, rows in rows_by_split.items()},
        "unique_inputs": len(unique_texts),
        "label_to_id": LABEL_TO_ID,
        "gigaevo_csv": str(csv_path.relative_to(OUT_DIR)).replace("\\", "/"),
        "gigaevo_rows": len(arranged_rows),
        "gigaevo_test_size": len(rows_by_split["validation"]) / len(arranged_rows),
        "gigaevo_split_assertion": "random_state=42 stratified split exactly equals dataset_3 validation",
        "source_sha256": {
            split: file_sha256(path) for split, path in SPLIT_PATHS.items()
        },
    }
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
