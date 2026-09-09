# Hello from MemoryLayer Integration Test

This is a test markdown document used to verify the full document ingestion
pipeline from data-connectors through MemoryLayer.

## Purpose

The local_fs connector discovers this file during a sync operation, computes
its SHA256 hash, uploads it to the blob store, and emits a `doc_added` Aether
pool task for MemoryLayer to process.

## Content for RAG Verification

The quick brown fox jumps over the lazy dog. This sentence contains every
letter of the English alphabet and is commonly used as a pangram for testing.

MemoryLayer should be able to find this document when searching for phrases
like "quick brown fox" or "pangram for testing" after ingestion completes.

## Technical Details

- File format: Markdown (text/markdown)
- Connector: local_fs
- Pipeline: sync -> doc_added -> fetch -> render -> embed -> store
