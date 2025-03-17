import logging
from enum import Enum
from typing import Any

import msgspec


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
    # Original local path
    path: str
    # Map [file-sha1, RepoFile]
    files_map: dict[str, RepoFile]
    # Map [commit-sha1, set[file-sha1]]
    commits_files_map: dict[str, set[str]]


class CodeRequestType(Enum):
    STOP = "stop"
    ADD_REPO = "add_repo"
    REMOVE_REPO = "remove_repo"
    QUERY_REPOS = "query_repos"


class CodeStatus(Enum):
    OK = "OK"
    ERROR = "ERROR"


class CodeRequest(msgspec.Struct):
    request: CodeRequestType
    payload: dict | None = None


class CodeReply(msgspec.Struct):
    status: CodeStatus
    message: str = ""
    payload: dict | None = None

    @staticmethod
    def from_error(message: str, logger: logging.Logger | None = None):
        if logger is not None:
            logger.error(message)
        return CodeReply(status=CodeStatus.ERROR, message=message)
