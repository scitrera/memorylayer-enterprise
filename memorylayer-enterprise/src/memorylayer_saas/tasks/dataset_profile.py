# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Task handler for dataset profile phase.

Phase 1 of the dataset pipeline:
- Mark dataset as PROFILING
- Profile the Parquet file with Polars (column stats, schema detection)
- Update dataset record with profile results
- Schedule dataset_summarize task
- Update job progress to 40%
"""
from datetime import datetime, timezone
from typing import Optional

from scitrera_app_framework import Variables, get_extension

from memorylayer_server.services.tasks.handlers import TaskHandlerPlugin
from memorylayer_server.services.tasks.base import TaskSchedule, EXT_TASK_SERVICE
from memorylayer_server.services.storage import EXT_STORAGE_BACKEND

from memorylayer_saas.models.dataset import DatasetStatus
from memorylayer_saas.services.dataset import get_dataset_service


class DatasetProfileTaskHandler(TaskHandlerPlugin):
    """Task handler for profiling uploaded datasets.

    Scans the Parquet file, computes column-level statistics, detects
    temporal columns, and persists the profile to the database.
    """

    def get_task_type(self) -> str:
        return "dataset_profile"

    async def handle(self, v: Variables, payload: dict) -> None:
        """Execute the profiling phase.

        Args:
            v: Variables instance.
            payload: Dict with dataset_id, job_id, workspace_id.
        """
        dataset_id = payload["dataset_id"]
        job_id = payload["job_id"]
        workspace_id = payload["workspace_id"]

        storage = get_extension(EXT_STORAGE_BACKEND, v)
        task_service = get_extension(EXT_TASK_SERVICE, v)
        dataset_service = get_dataset_service(v)

        try:
            now = datetime.now(timezone.utc)
            await storage.update_dataset(
                dataset_id,
                status=DatasetStatus.PROFILING.value,
                profiling_started_at=now,
            )
            await storage.update_dataset_job(
                job_id,
                status="running",
                started_at=now,
            )

            await dataset_service.profile_dataset(dataset_id, workspace_id)
            await storage.update_dataset_job(job_id, progress_percent=40)

            dataset_service.logger.info(
                "Profile phase complete for dataset %s", dataset_id,
            )

            # Check if summaries are requested
            ds = await storage.get_dataset(dataset_id, workspace_id)
            if ds and ds.profiling_options.generate_summaries:
                await task_service.schedule_task(
                    "dataset_summarize",
                    {
                        "dataset_id": dataset_id,
                        "job_id": job_id,
                        "workspace_id": workspace_id,
                    },
                    priority=3,
                )
            else:
                # Skip summarization, go straight to finalize
                await storage.update_dataset(
                    dataset_id,
                    status=DatasetStatus.COMPLETED.value,
                    profiling_completed_at=datetime.now(timezone.utc),
                )
                await storage.update_dataset_job(
                    job_id,
                    status="completed",
                    progress_percent=100,
                    datasets_processed=1,
                    completed_at=datetime.now(timezone.utc),
                )

        except Exception as exc:
            dataset_service.logger.error(
                "Profile phase failed for dataset %s: %s",
                dataset_id, exc, exc_info=True,
            )
            now = datetime.now(timezone.utc)
            await storage.update_dataset(
                dataset_id,
                status=DatasetStatus.FAILED.value,
                profiling_completed_at=now,
            )
            await storage.update_dataset_job(
                job_id,
                status="failed",
                completed_at=now,
                errors=[{"dataset_id": dataset_id, "error": str(exc)}],
            )

    def get_schedule(self, v: Variables) -> Optional[TaskSchedule]:
        return None
