import msgspec
from typing import Any


class Embeddings(msgspec.Struct):
    # Single or Unified embedding (mean pooled when chunks is not None)
    unified: Any
    # Per 1024 chunk embedding
    chunks: list[Any] | None


class RepoFile(msgspec.Struct):
    path: str
    raw: str
    summary: str | None
    embeddings: Embeddings | None


class RepoIndex(msgspec.Struct):
    files_map: dict[str, RepoFile]
    commits_files_map: dict[str, set[str]]
