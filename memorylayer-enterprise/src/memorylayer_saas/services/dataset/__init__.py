# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Dataset service package.

Provides extension points and plugin base classes for the dataset ingestion
and profiling service. Datasets (CSV, Parquet, JSON-lines, Excel) are uploaded,
profiled with column-level statistics, summarized by an LLM, and the summaries
are stored as memories for semantic retrieval.

Extension points:
- EXT_DATASET_SERVICE: Core dataset ingestion/profiling/slice orchestrator
"""
from scitrera_app_framework import Variables, get_extension
from scitrera_app_framework.api import Plugin, enabled_option_pattern

from ...config import (
    MEMORYLAYER_DATASET_SERVICE,
    DEFAULT_MEMORYLAYER_DATASET_SERVICE,
)

# Extension point constant
EXT_DATASET_SERVICE = 'memorylayer-enterprise-dataset-service'


# === Plugin Base Class ===

# noinspection PyAbstractClass
class DatasetServicePluginBase(Plugin):
    """Base plugin for dataset service."""
    PROVIDER_NAME: str = None

    def name(self) -> str:
        return f"{EXT_DATASET_SERVICE}|{self.PROVIDER_NAME}"

    def extension_point_name(self, v: Variables) -> str:
        return EXT_DATASET_SERVICE

    def is_enabled(self, v: Variables) -> bool:
        return enabled_option_pattern(
            self, v, MEMORYLAYER_DATASET_SERVICE, self_attr='PROVIDER_NAME'
        )

    def on_registration(self, v: Variables) -> None:
        v.set_default_value(
            MEMORYLAYER_DATASET_SERVICE,
            DEFAULT_MEMORYLAYER_DATASET_SERVICE,
        )


# === Convenience Getter ===

def get_dataset_service(v: Variables = None):
    """Get the dataset service instance."""
    return get_extension(EXT_DATASET_SERVICE, v)


__all__ = (
    'EXT_DATASET_SERVICE',
    'DatasetServicePluginBase',
    'get_dataset_service',
)
