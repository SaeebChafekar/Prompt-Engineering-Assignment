"""Ingestion: get a repository, filter it, parse Python, and cut it into chunks."""

from .models import Chunk, ClassInfo, FunctionInfo, ImportInfo, IngestResult, ParsedFile, SourceFile
from .pipeline import ingest_directory, ingest_repo

__all__ = [
    "Chunk",
    "ClassInfo",
    "FunctionInfo",
    "ImportInfo",
    "IngestResult",
    "ParsedFile",
    "SourceFile",
    "ingest_directory",
    "ingest_repo",
]
