"""Encryption utilities for sensitive data stored in the database.

Uses Fernet symmetric encryption (AES-128-CBC + HMAC-SHA256) for authenticated
encryption of JSON data such as DataProvider credentials.

The encrypted payload is stored in a JSONB column as
``{"_encrypted": "<fernet_token>"}``, which keeps the column type valid while
clearly distinguishing encrypted values from legacy plaintext dicts.

Configuration:
    Set the ``MEMORYLAYER_ENCRYPTION_KEY`` environment variable to a Fernet key.
    Generate one with::

        python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"

    If the key is not set, the module operates in **passthrough mode**: data is
    stored and returned as plain JSON dicts.  A warning is logged once on first use.
"""

import json
import logging
import os
from typing import Any, Optional

from cryptography.fernet import Fernet, InvalidToken

from ..config import MEMORYLAYER_ENCRYPTION_KEY

logger = logging.getLogger(__name__)

_fernet: Optional[Fernet] = None
_passthrough_warned: bool = False

# Sentinel key used inside the JSONB wrapper to indicate encrypted data
_ENCRYPTED_MARKER = "_encrypted"


def get_fernet() -> Optional[Fernet]:
    """Return a cached Fernet instance, or ``None`` when no key is configured."""
    global _fernet, _passthrough_warned

    if _fernet is not None:
        return _fernet

    key = os.environ.get(MEMORYLAYER_ENCRYPTION_KEY)
    if not key:
        if not _passthrough_warned:
            logger.warning(
                "%s is not set -- encrypted_args will be stored in plaintext (passthrough mode). "
                "Generate a key with: python -c \"from cryptography.fernet import Fernet; "
                "print(Fernet.generate_key().decode())\"",
                MEMORYLAYER_ENCRYPTION_KEY,
            )
            _passthrough_warned = True
        return None

    _fernet = Fernet(key.encode())
    return _fernet


def encrypt_json(data: dict[str, Any]) -> dict[str, Any]:
    """Encrypt a dict into a JSONB-compatible wrapper.

    Returns ``{"_encrypted": "<fernet_token>"}`` when a key is configured,
    or the original *data* dict unchanged in passthrough mode.
    """
    f = get_fernet()
    if f is None:
        return data
    plaintext = json.dumps(data, separators=(",", ":")).encode()
    token = f.encrypt(plaintext).decode()
    return {_ENCRYPTED_MARKER: token}


def decrypt_json(data: dict[str, Any]) -> dict[str, Any]:
    """Decrypt a JSONB value back to a plain dict.

    Handles three cases for backward compatibility:
    1. ``{"_encrypted": "<token>"}`` -- encrypted data, decrypt with Fernet.
    2. Any other dict -- legacy plaintext, returned as-is.
    3. Empty dict -- returned as-is.
    """
    if not data or _ENCRYPTED_MARKER not in data:
        # Legacy plaintext dict or empty -- return as-is
        return data

    token = data[_ENCRYPTED_MARKER]

    f = get_fernet()
    if f is None:
        raise ValueError(
            "Cannot decrypt data: %s is not set and the stored value is encrypted" % MEMORYLAYER_ENCRYPTION_KEY
        )

    try:
        plaintext = f.decrypt(token.encode())
        return json.loads(plaintext)
    except InvalidToken:
        raise ValueError("Failed to decrypt data: invalid token or wrong encryption key")
