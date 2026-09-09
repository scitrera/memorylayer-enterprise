"""Request/response message shapes for data-connectors HTTP surface.

Uses Pydantic BaseModel to match MemoryLayer's existing API conventions.
Field types are kept runtime-agnostic for potential Go v2 re-derivation.
"""
