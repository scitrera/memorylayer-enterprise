"""
Enterprise Memory domain model with multi-vector (ColPali) support.

Extends the OSS Memory model to add multi-vector embedding capabilities
for late interaction retrieval (MaxSim scoring).
"""
from typing import Optional

from pydantic import Field

from memorylayer_server.models.memory import Memory as MemoryBase


class Memory(MemoryBase):
    """
    Enterprise Memory with multi-vector embedding support.

    Extends the OSS Memory model to add ColPali multi-vector embeddings
    for late interaction (MaxSim) retrieval, which is particularly effective
    for documents with visual elements like PDFs.
    """

    # Multi-vector embedding for ColPali (array of 128-dim vectors for late interaction)
    multivector: Optional[list[list[float]]] = Field(
        None,
        description="Multi-vector embedding for MaxSim retrieval (ColPali)"
    )
