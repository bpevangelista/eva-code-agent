import os
import hashlib
import lz4.frame
import msgspec

from schemas import RepoIndex, RepoFile, RepoBlobEmbeddings
from git import Blob, Repo, Head
from embedding_model import EmbeddingModel

from platform_utils import get_data_path
from logging_config import get_logger

logger = get_logger(__name__)

APP_ID = "evacode"
embedding_model = None

# In-memory index maps
global_index_map: dict[str, RepoIndex] = {}


def get_repo_uuid(repo: Repo) -> str:
    # TODO Support repos without a remote
    remote_url = repo.remotes.origin.url.strip()
    parts = remote_url.split(":")[-1].split(".git")[0].split("/")
    repo_path = f"{parts[-2]}_{parts[-1]}"
    return f"{repo_path}_{hashlib.sha256(remote_url.encode()).hexdigest()[:16]}"


def get_repo_files(repo_branch: Head, exclude_extensions: list[str] | None = None):
    logger.info(f"Traversing repo files: {repo_branch.name}")
    repo_commit = repo_branch.commit
    files_map: dict[str, RepoFile] = {}
    commits_files_set: set[str] = set()

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

    return files_map, commits_files_set


def load_or_create_repo_index(repo_uuid: str) -> RepoIndex:
    app_data_path = get_data_path(APP_ID)
    repo_index_path = os.path.join(app_data_path, f"{repo_uuid}.lz4")
    logger.info(f"Loading index: {repo_index_path}")

    if os.path.exists(repo_index_path):
        try:
            with lz4.frame.open(repo_index_path, "rb") as f:
                return msgspec.msgpack.decode(f.read(), type=RepoIndex)
        except (OSError, msgspec.DecodeError) as e:
            logger.error(f"Failed reading: {repo_index_path}\n{e}")

    return RepoIndex(uuid=repo_uuid, files_map={}, commits_files_map={})


def save_repo_index(repo_index: RepoIndex):
    app_data_path = get_data_path(APP_ID)
    repo_index_path = os.path.join(app_data_path, f"{repo_index.uuid}.lz4")
    logger.info(f"Saving index: {repo_index_path}")

    try:
        encoded_data = msgspec.msgpack.encode(repo_index)
        with lz4.frame.open(repo_index_path, "wb") as f:
            f.write(encoded_data)
    except OSError as e:
        print(f"Failed saving: {repo_index_path}\n{e}")


def generate_embeddings(repo_index: RepoIndex):
    global embedding_model
    logger.info(f"Generating embeddings: {repo_index.uuid}")

    count = 0
    for key, file in repo_index.files_map.items():
        if file.embeddings is None:
            embedding_model = embedding_model or EmbeddingModel()
            logger.info(f"  Embeddings: {file.path}")
            result_embedding = embedding_model.generate([file.raw])
            file.embeddings = RepoBlobEmbeddings(unified=result_embedding)
            repo_index.files_map[key] = file

        count += 1
        if count > 10000:
            break


# User provided
seed_repo = "~/projects/vfastml"

repo = Repo(seed_repo, search_parent_directories=True)
repo_uuid = get_repo_uuid(repo)

repo_index: RepoIndex = load_or_create_repo_index(repo_uuid)
files_map, commits_files_set = get_repo_files(repo.active_branch)

# Merge repo_index, files_map, commits_files_set
if repo.active_branch.commit.hexsha not in repo_index.commits_files_map:
    repo_index.commits_files_map[repo.active_branch.commit.hexsha] = commits_files_set
    repo_index.files_map.update({k: v for k, v in files_map.items() if k not in repo_index.files_map})
    save_repo_index(repo_index)

global_index_map[repo_index.uuid] = repo_index
generate_embeddings(repo_index)
save_repo_index(repo_index)

del repo
