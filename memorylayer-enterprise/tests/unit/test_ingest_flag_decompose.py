# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Per-document decomposition opt-out, declared at upload.

Suppresses only the fan-out. The composite page memory is still created, so
`has_any_memory` stays true and gap analysis is untouched -- a document with
decomposition off is still legitimately COMPLETE, and doc_verify will not
re-drive it forever.

The flag is a TRI-STATE on the wire: absent means "not specified" and must fall
back to the global default, never read as a decision to disable.
"""

from unittest.mock import MagicMock

from memorylayer_saas.models.document import Document, DocumentType
from memorylayer_saas.services.document.gap_analysis import (
    EffectiveFlags,
    resolve_effective_flags,
)


def _v(decompose_env: bool = True):
    v = MagicMock()
    v.environ = MagicMock(
        side_effect=lambda key, default=None, **kw: (
            decompose_env if "FACT_DECOMPOSITION_ENABLED" in key else default
        )
    )
    return v


def _doc(ingest_flags=None) -> Document:
    return Document(
        id="doc_1", workspace_id="ws", filename="a.pdf",
        document_type=DocumentType.PDF, content_hash="d" * 64, size_bytes=1,
        metadata={"ingest_flags": ingest_flags} if ingest_flags is not None else {},
    )


class TestDecomposeFlagResolution:
    def test_absent_flag_falls_back_to_the_global_default(self):
        # The tri-state. A document pinned before this flag existed has no key,
        # and must keep decomposing rather than silently stop.
        flags = resolve_effective_flags(_v(True), _doc({"transcribe": True}))
        assert flags.decompose is True

    def test_absent_flag_honours_a_disabled_global(self):
        flags = resolve_effective_flags(_v(False), _doc({"transcribe": True}))
        assert flags.decompose is False

    def test_explicit_false_disables_for_this_document_only(self):
        flags = resolve_effective_flags(_v(True), _doc({"decompose": False}))
        assert flags.decompose is False

    def test_explicit_true_overrides_a_disabled_global(self):
        flags = resolve_effective_flags(_v(False), _doc({"decompose": True}))
        assert flags.decompose is True

    def test_unpinned_document_uses_the_environment(self):
        assert resolve_effective_flags(_v(False), _doc()).decompose is False
        assert resolve_effective_flags(_v(True), _doc()).decompose is True

    def test_an_explicit_flags_argument_still_wins(self):
        override = EffectiveFlags(
            transcribe=True, chat_ingest=False, visual_tokenizer=False, decompose=False,
        )
        assert resolve_effective_flags(_v(True), _doc(), flags=override).decompose is False

    def test_the_flag_is_pinned_so_it_survives_resume(self):
        # as_dict is what finalize writes into metadata['ingest_flags'];
        # a gap-fill re-run must make the same decision as the first pass.
        pinned = EffectiveFlags(
            transcribe=True, chat_ingest=False, visual_tokenizer=False, decompose=False,
        ).as_dict()
        assert pinned["decompose"] is False
        assert resolve_effective_flags(_v(True), _doc(pinned)).decompose is False

    def test_other_flags_are_unaffected(self):
        flags = resolve_effective_flags(_v(True), _doc({"decompose": False}))
        assert flags.transcribe is True
        assert flags.chat_ingest is False


def _doc_md(md) -> Document:
    return Document(
        id="doc_1", workspace_id="ws", filename="a.pdf",
        document_type=DocumentType.PDF, content_hash="d" * 64, size_bytes=1,
        metadata=md or {},
    )


class TestUploadRequestedFlags:
    """`requested_ingest_flags` — what the uploader asked for, seeded at document
    CREATION from the VFS entry.

    Kept under its own key rather than seeding `ingest_flags` directly:
    `ingest_flags` is written at finalize as a COMPLETE set, and its branch reads
    absent keys as HARDCODED defaults (transcribe=True, chat_ingest=False,
    visual_tokenizer=False). Seeding a partial dict there would silently freeze
    those three at values the deployment never configured. These tests pin that
    separation, because the failure would be invisible: ingestion would simply
    run with flags nobody chose.
    """

    def test_request_beats_the_environment_default(self):
        flags = resolve_effective_flags(
            _v(True), _doc_md({"requested_ingest_flags": {"decompose": False}}))
        assert flags.decompose is False

    def test_request_can_also_enable_against_a_disabled_global(self):
        flags = resolve_effective_flags(
            _v(False), _doc_md({"requested_ingest_flags": {"decompose": True}}))
        assert flags.decompose is True

    def test_a_finalize_pin_outranks_the_request(self):
        # Once pinned, a gap-fill re-run must repeat the FIRST pass's decision
        # rather than re-derive it from the upload request.
        flags = resolve_effective_flags(_v(True), _doc_md({
            "requested_ingest_flags": {"decompose": True},
            "ingest_flags": {"decompose": False},
        }))
        assert flags.decompose is False

    def test_a_legacy_pin_without_the_key_still_honours_the_request(self):
        # Pinned before the flag existed: the pinned set has no `decompose`, so
        # the request supplies the fallback instead of the env default.
        flags = resolve_effective_flags(_v(True), _doc_md({
            "requested_ingest_flags": {"decompose": False},
            "ingest_flags": {"transcribe": True},
        }))
        assert flags.decompose is False

    def test_requesting_decompose_does_not_disturb_the_other_flags(self):
        """The whole reason for the separate key. Requesting only `decompose`
        must leave transcribe/chat_ingest/visual_tokenizer resolving from env."""
        v = MagicMock()
        v.environ = MagicMock(side_effect=lambda key, default=None, **kw: (
            False if "FACT_DECOMPOSITION_ENABLED" in key
            else True  # every other flag ON in this deployment
        ))
        flags = resolve_effective_flags(
            v, _doc_md({"requested_ingest_flags": {"decompose": True}}))
        assert flags.decompose is True
        assert flags.transcribe is True
        assert flags.chat_ingest is True
        assert flags.visual_tokenizer is True

    def test_a_malformed_request_is_ignored(self):
        # Entry metadata is written by whoever minted the upload; junk must not
        # crash ingestion.
        for junk in ("nonsense", 42, [], None):
            flags = resolve_effective_flags(
                _v(True), _doc_md({"requested_ingest_flags": junk}))
            assert flags.decompose is True
