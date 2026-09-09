# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Request/response message shapes for data-connectors HTTP surface.

Uses Pydantic BaseModel to match MemoryLayer's existing API conventions.
Field types are kept runtime-agnostic for potential Go v2 re-derivation.
"""
