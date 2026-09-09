# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Office / HTML -> PDF conversion via LibreOffice (``soffice``).

HTML, DOCX and PPTX uploads are converted to PDF so they flow through the
same page/visual rendering pipeline as native PDFs (``pdf2image`` -> PNG ->
embed/transcribe).  Conversion is delegated to a headless LibreOffice
process (``soffice --headless --convert-to pdf``).

Dependency note:
    This requires a LibreOffice install providing the ``soffice`` binary on
    ``PATH`` (or at ``MEMORYLAYER_SOFFICE_BINARY``).  It is an OS-level
    dependency, not a Python package.  When the binary is absent or the
    conversion fails, :class:`OfficeConversionError` is raised with an
    actionable message so the ingestion job fails at the render stage with a
    useful error rather than a bare ``NotImplementedError``.

    Currently no container image ships LibreOffice; the binary must be added
    to the deployment environment.  Until then every office/HTML document will
    fail at the render stage with an OfficeConversionError naming the missing
    binary and the ``MEMORYLAYER_SOFFICE_BINARY`` override path.
"""
from __future__ import annotations

import asyncio
import glob
import os
import shutil
import subprocess
import tempfile

# Default filename stem written to the temp working dir before conversion.
_INPUT_STEM = "input"

# Environment override for the LibreOffice binary location.
_SOFFICE_BINARY_ENV = "MEMORYLAYER_SOFFICE_BINARY"

# Conversion timeout in seconds.  A hung soffice (malformed input, profile-lock
# contention) would otherwise block a thread-pool slot indefinitely.
_SOFFICE_TIMEOUT_ENV = "MEMORYLAYER_SOFFICE_TIMEOUT"
_DEFAULT_SOFFICE_TIMEOUT = 120

# Extensions we know how to hand to LibreOffice.  Keyed by the lowercase
# extension (no leading dot) so callers can map a DocumentType to a suffix.
SUPPORTED_OFFICE_EXTENSIONS = ("html", "htm", "docx", "pptx", "txt")


class OfficeConversionError(RuntimeError):
    """Raised when an office/HTML document cannot be converted to PDF."""


def _resolve_soffice_binary() -> str | None:
    """Return the path to the ``soffice`` binary, or ``None`` if unavailable.

    Honors the ``MEMORYLAYER_SOFFICE_BINARY`` override first, then falls back
    to ``soffice`` / ``libreoffice`` on ``PATH``.
    """
    override = os.environ.get(_SOFFICE_BINARY_ENV)
    if override:
        return override if os.path.isfile(override) else shutil.which(override)
    for candidate in ("soffice", "libreoffice"):
        found = shutil.which(candidate)
        if found:
            return found
    return None


def _resolve_timeout() -> int:
    """Return the configured soffice timeout in seconds."""
    try:
        return int(os.environ.get(_SOFFICE_TIMEOUT_ENV, _DEFAULT_SOFFICE_TIMEOUT))
    except (TypeError, ValueError):
        return _DEFAULT_SOFFICE_TIMEOUT


def _convert_office_bytes_to_pdf(data: bytes, source_ext: str) -> bytes:
    """Convert raw office/HTML bytes to PDF bytes (blocking).

    Each invocation runs in its own isolated LibreOffice user-profile directory
    (``-env:UserInstallation=file://<workdir>/lo-profile``) so concurrent
    conversions do not collide on the shared ``~/.config/libreoffice`` lock.

    Args:
        data: The raw uploaded file bytes.
        source_ext: Lowercase source extension without a leading dot
            (e.g. ``"docx"``).

    Returns:
        The converted PDF as bytes.

    Raises:
        OfficeConversionError: If ``soffice`` is unavailable, times out, exits
            non-zero, or produces no PDF output.
    """
    soffice = _resolve_soffice_binary()
    if soffice is None:
        raise OfficeConversionError(
            "LibreOffice (soffice) is not available; cannot convert .%s "
            "documents to PDF. Install LibreOffice or set %s to the binary "
            "path." % (source_ext, _SOFFICE_BINARY_ENV)
        )

    timeout = _resolve_timeout()

    with tempfile.TemporaryDirectory(prefix="ml-office-convert-") as workdir:
        input_path = os.path.join(workdir, "%s.%s" % (_INPUT_STEM, source_ext))
        with open(input_path, "wb") as fh:
            fh.write(data)

        # Isolated user profile: avoids lock contention when multiple
        # conversions run concurrently (each temp dir is unique per call).
        profile_url = "file://%s/lo-profile" % workdir

        try:
            proc = subprocess_run(
                [
                    soffice,
                    "--headless",
                    "-env:UserInstallation=%s" % profile_url,
                    "--convert-to",
                    "pdf",
                    "--outdir",
                    workdir,
                    input_path,
                ],
                cwd=workdir,
                timeout=timeout,
            )
        except FileNotFoundError as exc:
            raise OfficeConversionError(
                "LibreOffice binary %r could not be executed: %s" % (soffice, exc)
            ) from exc
        except subprocess.TimeoutExpired as exc:
            # subprocess.TimeoutExpired subclasses SubprocessError (NOT the builtin
            # TimeoutError), so it must be caught explicitly.
            raise OfficeConversionError(
                "LibreOffice conversion of .%s timed out after %d seconds"
                % (source_ext, timeout)
            ) from exc

        if proc.returncode != 0:
            raise OfficeConversionError(
                "LibreOffice conversion of .%s failed (exit %d): %s"
                % (source_ext, proc.returncode, _decode(proc.stderr) or _decode(proc.stdout))
            )

        # Prefer the deterministic expected output path; fall back to glob
        # in case soffice uses a slightly different stem on some platforms.
        expected_pdf = os.path.join(workdir, "%s.pdf" % _INPUT_STEM)
        if os.path.isfile(expected_pdf):
            pdf_path = expected_pdf
        else:
            produced = glob.glob(os.path.join(workdir, "*.pdf"))
            if not produced:
                raise OfficeConversionError(
                    "LibreOffice produced no PDF output for the .%s document" % source_ext
                )
            pdf_path = produced[0]

        with open(pdf_path, "rb") as fh:
            return fh.read()


def subprocess_run(args: list[str], cwd: str, timeout: int = _DEFAULT_SOFFICE_TIMEOUT):
    """Run ``args`` capturing output.

    Isolated in a tiny wrapper so tests can patch the subprocess call without
    spawning a real LibreOffice process.

    Args:
        args: Command and arguments.
        cwd: Working directory for the subprocess.
        timeout: Seconds before ``subprocess.TimeoutExpired`` is raised.

    Raises:
        subprocess.TimeoutExpired: When the process does not finish within
            ``timeout`` seconds.  The caller (``_convert_office_bytes_to_pdf``)
            converts this to :class:`OfficeConversionError`.
    """
    import subprocess

    return subprocess.run(
        args,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )


def _decode(value) -> str:
    """Best-effort decode of subprocess output bytes to a trimmed string."""
    if not value:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    return str(value).strip()


async def convert_office_bytes_to_pdf(data: bytes, source_ext: str) -> bytes:
    """Async wrapper running the blocking LibreOffice conversion off-loop.

    Args:
        data: The raw uploaded file bytes.
        source_ext: Lowercase source extension without a leading dot.

    Returns:
        The converted PDF as bytes.

    Raises:
        OfficeConversionError: On any conversion failure (see
            :func:`_convert_office_bytes_to_pdf`).
    """
    return await asyncio.to_thread(_convert_office_bytes_to_pdf, data, source_ext)
