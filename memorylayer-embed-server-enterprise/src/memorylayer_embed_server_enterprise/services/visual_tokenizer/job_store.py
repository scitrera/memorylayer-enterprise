"""Persistent async visual-tokenizer job result storage."""

import json
import uuid
from pathlib import Path

from .base import ExtractionResult, VisualFeatureMetadata, VisualFeatures

_METADATA_ONLY_RETENTION = "metadata_only_until_job_ttl"
_METADATA_AND_TENSOR_RETENTION = "metadata_and_tensors_until_job_ttl"


def _feature_metadata_from_features(features: VisualFeatures) -> VisualFeatureMetadata:
    return VisualFeatureMetadata(
        image_grid_thw=features.image_grid_thw,
        num_visual_tokens=features.num_visual_tokens,
        hidden_dim=features.hidden_dim,
        dtype=features.dtype,
        original_image_size=features.original_image_size,
        embed_kind=features.embed_kind,
    )


def _feature_metadata_from_result(result: ExtractionResult) -> VisualFeatureMetadata | None:
    if result.features is not None:
        return _feature_metadata_from_features(result.features)
    return result.feature_metadata


def _metadata_to_payload(metadata: VisualFeatureMetadata) -> dict:
    return {
        "image_grid_thw": metadata.image_grid_thw,
        "num_visual_tokens": metadata.num_visual_tokens,
        "hidden_dim": metadata.hidden_dim,
        "dtype": metadata.dtype,
        "original_image_size": list(metadata.original_image_size),
        "embed_kind": metadata.embed_kind,
    }


def _metadata_from_payload(payload: dict) -> VisualFeatureMetadata:
    original_size = payload["original_image_size"]
    return VisualFeatureMetadata(
        image_grid_thw=list(payload["image_grid_thw"]),
        num_visual_tokens=int(payload["num_visual_tokens"]),
        hidden_dim=int(payload["hidden_dim"]),
        dtype=str(payload["dtype"]),
        original_image_size=(int(original_size[0]), int(original_size[1])),
        embed_kind=str(payload.get("embed_kind", "image_embeds")),
    )


def _artifact_retention(tensors_persisted: bool) -> str:
    if tensors_persisted:
        return _METADATA_AND_TENSOR_RETENTION
    return _METADATA_ONLY_RETENTION


class VisualTokenizerJobResultStore:
    """Stores completed async job results outside the in-memory job table."""

    def __init__(self, root_dir: str | Path):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def put_status(self, job_id: str, job: dict) -> str:
        """Persist lightweight job status without loading or writing tensors."""
        job_dir = self._job_dir(job_id)
        job_dir.mkdir(parents=True, exist_ok=True)
        payload = self._read_payload(job_id) or {"results": []}
        payload["job"] = job
        self._write_payload(job_dir, payload)
        return str(job_dir)

    def put(
        self,
        job_id: str,
        results: list[ExtractionResult],
        *,
        return_tensors: bool,
        created_at: float,
        updated_at: float,
    ) -> str:
        """Persist completed job results and return the job directory.

        Async job retention follows the caller's response-shape request:
        ``return_tensors=True`` stores tensor artifacts until the job TTL/delete
        path removes the job directory, while ``return_tensors=False`` stores
        only restartable metadata and never writes per-result safetensors files.
        The visual-tokenizer cache is a separate artifact store governed by its
        own cache TTL/size policy.
        """
        job_dir = self._job_dir(job_id)
        job_dir.mkdir(parents=True, exist_ok=True)
        entries: list[dict] = []
        tensors_persisted = False
        save_file = None

        for result in results:
            feature_metadata = _feature_metadata_from_result(result)
            feature_meta = (
                _metadata_to_payload(feature_metadata)
                if result.success and feature_metadata is not None
                else None
            )
            tensor_file = None
            if return_tensors and result.success and result.features is not None:
                if save_file is None:
                    from safetensors.torch import save_file as safetensors_save_file

                    save_file = safetensors_save_file
                tensor_file = f"{result.image_index}.safetensors"
                save_file(
                    {"image_embeds": result.features.image_embeds},
                    str(job_dir / tensor_file),
                )
                tensors_persisted = True

            entries.append({
                "image_index": result.image_index,
                "success": result.success,
                "error": result.error,
                "latency_ms": result.latency_ms,
                "from_cache": result.from_cache,
                "cache_path": result.cache_path,
                "features": feature_meta,
                "tensor_file": tensor_file,
                "tensors_persisted": tensor_file is not None,
            })

        payload = {
            "job": {
                "status": "completed",
                "progress": len(results),
                "total": len(results),
                "result": None,
                "result_ref": str(job_dir),
                "error": None,
                "created_at": created_at,
                "updated_at": updated_at,
                "return_tensors": return_tensors,
                "tensors_persisted": tensors_persisted,
                "artifact_retention": _artifact_retention(tensors_persisted),
            },
            "results": entries,
        }
        self._write_payload(job_dir, payload)
        return str(job_dir)

    def get_status(self, job_id: str) -> dict | None:
        """Load completed job metadata without loading tensor results."""
        payload = self._read_payload(job_id)
        if payload is None:
            return None
        job = payload.get("job")
        if not isinstance(job, dict):
            self.delete(job_id)
            return None
        return job

    def get(self, job_id: str) -> list[ExtractionResult] | None:
        """Load persisted results for a completed job."""
        try:
            payload = self._read_payload(job_id)
            if payload is None:
                return None

            job_dir = self._job_dir(job_id)
            results = []
            load_file = None
            for entry in payload["results"]:
                features = None
                feature_metadata = None
                feature_meta = entry.get("features")
                tensor_file = entry.get("tensor_file")
                if feature_meta is not None:
                    feature_metadata = _metadata_from_payload(feature_meta)
                    if tensor_file is not None:
                        if load_file is None:
                            from safetensors.torch import load_file as safetensors_load_file

                            load_file = safetensors_load_file
                        tensors = load_file(str(job_dir / tensor_file))
                        features = VisualFeatures(
                            image_embeds=tensors["image_embeds"],
                            image_grid_thw=feature_metadata.image_grid_thw,
                            num_visual_tokens=feature_metadata.num_visual_tokens,
                            hidden_dim=feature_metadata.hidden_dim,
                            dtype=feature_metadata.dtype,
                            original_image_size=feature_metadata.original_image_size,
                            embed_kind=feature_metadata.embed_kind,
                        )

                results.append(
                    ExtractionResult(
                        image_index=int(entry["image_index"]),
                        success=bool(entry["success"]),
                        features=features,
                        feature_metadata=feature_metadata,
                        error=entry.get("error"),
                        latency_ms=float(entry.get("latency_ms", 0.0)),
                        from_cache=bool(entry.get("from_cache", False)),
                        cache_path=entry.get("cache_path"),
                    )
                )
            return results
        except (IndexError, KeyError, TypeError, ValueError, OSError):
            return None

    def delete(self, job_id: str) -> None:
        """Delete persisted results for a job."""
        try:
            job_dir = self._job_dir(job_id)
        except ValueError:
            return
        if not job_dir.exists():
            return
        for path in job_dir.iterdir():
            path.unlink(missing_ok=True)
        job_dir.rmdir()

    def _read_payload(self, job_id: str) -> dict | None:
        try:
            metadata_path = self._job_dir(job_id) / "results.json"
        except ValueError:
            return None
        if not metadata_path.exists():
            return None
        try:
            with metadata_path.open() as file:
                payload = json.load(file)
        except (json.JSONDecodeError, OSError):
            return None
        if not isinstance(payload, dict):
            return None
        return payload

    def _write_payload(self, job_dir: Path, payload: dict) -> None:
        tmp_path = job_dir / "results.json.tmp"
        final_path = job_dir / "results.json"
        with tmp_path.open("w") as file:
            json.dump(payload, file, separators=(",", ":"))
        tmp_path.replace(final_path)

    def _job_dir(self, job_id: str) -> Path:
        safe_job_id = str(uuid.UUID(job_id))
        return self.root_dir / safe_job_id
