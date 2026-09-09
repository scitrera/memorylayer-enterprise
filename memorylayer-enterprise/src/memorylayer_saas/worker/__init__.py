# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""MemoryLayer Worker - distributed task worker process.

Connects to Aether as an Agent and executes MemoryLayer task handlers
received via the messaging system.

Usage:
    memorylayer-worker start --aether-addr localhost:50051
"""

from .runner import WorkerRunner

__all__ = ("WorkerRunner",)
