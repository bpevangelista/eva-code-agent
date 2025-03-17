import hashlib

from git import Blob, Repo

from logging_config import get_logger
from schemas import RepoFile

logger = get_logger(__name__)


def get_repo(repo_path: str) -> Repo:
    return Repo(repo_path, search_parent_directories=True)


def get_repo_uuid(repo: Repo) -> str:
    # TODO Support repos without a remote
    remote_url = repo.remotes.origin.url.strip()
    parts = remote_url.split(":")[-1].split(".git")[0].split("/")
    repo_path = f"{parts[-2]}_{parts[-1]}"
    return f"{repo_path}_{hashlib.sha256(remote_url.encode()).hexdigest()[:16]}"


def get_repo_files(repo: Repo, exclude_extensions: list[str] | None = None) -> (dict[str, RepoFile], set[str]):
    repo_branch = repo.active_branch
    logger.info(f'Traversing branch: "{repo.active_branch}" on {repo.git_dir}')
    files_map: dict[str, RepoFile] = {}
    commits_files_set: set[str] = set()

    repo_commit = repo_branch.commit
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
        commits_files_set.add(blob.hexsha)

    logger.info(f"  {len(files_map)} files traversed")
    return files_map, commits_files_set
