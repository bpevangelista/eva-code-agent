from schemas import RepoIndex, RepoFile
from git import Blob, Repo, Head
from embedding_model import EmbeddingModel

from logging_config import get_logger

logger = get_logger(__name__)

embedding_model = EmbeddingModel()
_global_repo_index: RepoIndex


def get_repo_files(repo_branch: Head, exclude_extensions: list[str] | None = None):
    repo_commit = repo_branch.commit
    files_map: dict[str, RepoFile] = {}
    commits_files_map: dict[str, set[str]] = {repo_commit.hexsha: set()}

    for blob in repo_commit.tree.traverse():
        if not isinstance(blob, Blob) or not blob.mime_type.startswith("text/"):
            continue
        if exclude_extensions and any(blob.path.endswith(ext) for ext in exclude_extensions):
            continue

        try:
            file_raw = blob.data_stream.read().decode("utf-8")
        # A binary file? Silently skip
        except UnicodeDecodeError:
            continue
        except Exception:
            logger.warning(f"Failed to read: {blob.path}. Skipping...")
            continue

        files_map[blob.hexsha] = RepoFile(path=blob.path, raw=file_raw, summary=None, embeddings=None)
        commits_files_map[repo_commit.hexsha].add(blob.hexsha)

    return files_map, commits_files_map


repo = Repo("../../vfastml", search_parent_directories=True)
files_map, commits_set = get_repo_files(repo.active_branch)

count = 0
for key, file in files_map.items():
    file.embeddings = embedding_model.generate([file.raw])

    count += 1
    if count > 10000:
        break

del repo
