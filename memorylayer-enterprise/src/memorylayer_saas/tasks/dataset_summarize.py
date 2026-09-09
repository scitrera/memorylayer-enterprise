"""Task handler for dataset summarize + embed phase.

Phase 2 of the dataset pipeline:
- Mark dataset as SUMMARIZING
- Generate LLM natural-language summary from the profile
- Embed summaries and store as memories
- Finalize dataset and job status
"""
from datetime import datetime, timezone
from typing import Optional

from scitrera_app_framework import Variables, get_extension

from memorylayer_server.services.tasks.handlers import TaskHandlerPlugin
from memorylayer_server.services.tasks.base import TaskSchedule
from memorylayer_server.services.storage import EXT_STORAGE_BACKEND

from memorylayer_saas.models.dataset import DatasetStatus
from memorylayer_saas.services.dataset import get_dataset_service


class DatasetSummarizeTaskHandler(TaskHandlerPlugin):
    """Task handler for summarizing and embedding dataset profiles.

    Generates LLM summaries from the statistical profile, embeds them,
    and stores them as memories for semantic retrieval.
    """

    def get_task_type(self) -> str:
        return "dataset_summarize"

    async def handle(self, v: Variables, payload: dict) -> None:
        """Execute the summarize + embed phase.

        Args:
            v: Variables instance.
            payload: Dict with dataset_id, job_id, workspace_id.
        """
        dataset_id = payload["dataset_id"]
        job_id = payload["job_id"]
        workspace_id = payload["workspace_id"]

        storage = get_extension(EXT_STORAGE_BACKEND, v)
        dataset_service = get_dataset_service(v)

        try:
            await storage.update_dataset(
                dataset_id,
                status=DatasetStatus.SUMMARIZING.value,
            )
            await storage.update_dataset_job(job_id, progress_percent=50)

            # Phase 3: LLM summarization
            await dataset_service.summarize_dataset(dataset_id, workspace_id)
            await storage.update_dataset_job(job_id, progress_percent=70)

            # Phase 4: Embed and store memories
            memory_ids = await dataset_service.embed_and_store_memories(
                dataset_id, workspace_id,
            )
            await storage.update_dataset_job(job_id, progress_percent=90)

            # Phase 5: Finalize
            now = datetime.now(timezone.utc)
            final_status = DatasetStatus.COMPLETED if memory_ids else DatasetStatus.FAILED

            await storage.update_dataset(
                dataset_id,
                status=final_status.value,
                memory_ids=memory_ids,
                profiling_completed_at=now,
            )
            await storage.update_dataset_job(
                job_id,
                status="completed",
                progress_percent=100,
                datasets_processed=1,
                total_memories_created=len(memory_ids),
                completed_at=now,
            )

            dataset_service.logger.info(
                "Summarize phase complete for dataset %s: %d memories created",
                dataset_id, len(memory_ids),
            )

        except Exception as exc:
            dataset_service.logger.error(
                "Summarize phase failed for dataset %s: %s",
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
