import msgspec
from typing import Any


class RepoBlobEmbeddings(msgspec.Struct):
    # Single or Unified embedding (mean pooled when chunks is not None)
    unified: Any
    # Per 1024 chunk embedding
    chunks: list[Any] | None = None


class RepoFile(msgspec.Struct):
    path: str
    raw: str
    summary: str | None
    embeddings: RepoBlobEmbeddings | None


class RepoIndex(msgspec.Struct):
    uuid: str
    # Map [file-sha1, RepoFile]
    files_map: dict[str, RepoFile]
    # Map [commit-sha1, set[file-sha1]]
    commits_files_map: dict[str, set[str]]
