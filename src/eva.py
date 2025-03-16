import os
import tempfile

import click
import lz4.frame
import msgspec
import zmq
from git import Repo

from embedding_model import EmbeddingModel
from git_utils import get_repo, get_repo_files, get_repo_uuid
from logging_config import get_logger
from platform_utils import get_data_path
from schemas import (
    CodeReply,
    CodeRequest,
    CodeRequestType,
    CodeStatus,
    RepoBlobEmbeddings,
    RepoIndex,
)

APP_ID = "evacode"
logger = get_logger(APP_ID)



class EvaCode:
    REPO_MAX_INDEXED_FILES = 1024 * 1024
    REPO_CHECKPOINT_TRIGGER_COUNT = 100

    def __init__(self):
        # In-memory index maps
        self._repo_index_map: dict[str, RepoIndex] = {}
        self.embedding_model = EmbeddingModel()

    @staticmethod
    def _load_or_create_repo_index(repo: Repo) -> RepoIndex:
        repo_uuid = get_repo_uuid(repo)
        app_data_path = get_data_path(APP_ID)
        repo_index_path = os.path.join(app_data_path, f"{repo_uuid}.lz4")
        logger.info(f"Loading index: {repo_index_path}")

        if os.path.exists(repo_index_path):
            try:
                with lz4.frame.open(repo_index_path, "rb") as f:
                    return msgspec.msgpack.decode(f.read(), type=RepoIndex)
            except (OSError, msgspec.DecodeError) as e:
                logger.error(f"Failed reading: {repo_index_path}\n{e}")

        repo_path = os.path.dirname(repo.git_dir)
        return RepoIndex(uuid=repo_uuid, path=repo_path, files_map={}, commits_files_map={})

    @staticmethod
    def _save_repo_index(repo_index: RepoIndex):
        app_data_path = get_data_path(APP_ID)
        repo_index_path = os.path.join(app_data_path, f"{repo_index.uuid}.lz4")
        logger.info(f"Saving index: {repo_index_path}")

        try:
            encoded_data = msgspec.msgpack.encode(repo_index)
            with lz4.frame.open(repo_index_path, "wb") as f:
                f.write(encoded_data)
        except OSError as e:
            print(f"Failed saving: {repo_index_path}\n{e}")

    def _generate_repo_embeddings(self, repo_index: RepoIndex):
        logger.info(f"Embedding repo: {repo_index.path}")

        if not any(repo_file.embeddings is None for repo_file in repo_index.files_map.values()):
            logger.info(f"  Skip, already done")

        embedding_count = 0
        files_count = 0
        for key, file in repo_index.files_map.items():
            if file.embeddings is None:
                try:
                    logger.info(f"  Embeddings: {file.path}")
                    result_embedding = self.embedding_model.generate([file.raw])
                    file.embeddings = RepoBlobEmbeddings(unified=result_embedding)
                    repo_index.files_map[key] = file
                    embedding_count += 1
                except Exception as e:
                    logger.error(f"Failed embedding: {file.path} {e}")

            files_count += 1
            if embedding_count >= EvaCode.REPO_CHECKPOINT_TRIGGER_COUNT:
                EvaCode._save_repo_index(repo_index)
                embedding_count = 0
            if files_count >= EvaCode.REPO_MAX_INDEXED_FILES:
                break

        if embedding_count > 0:
            EvaCode._save_repo_index(repo_index)

    def add_repo_to_index(self, repo_path: str) -> bool:
        logger.info(f"Adding to index: {repo_path}")
        try:
            repo = get_repo(repo_path)
            repo_index: RepoIndex = EvaCode._load_or_create_repo_index(repo)
            files_map, commits_files_set = get_repo_files(repo)

            # Merge repo_index, files_map, commits_files_set
            if repo.active_branch.commit.hexsha not in repo_index.commits_files_map:
                logger.info(f"  Updating index: {repo_path}")
                repo_index.commits_files_map[repo.active_branch.commit.hexsha] = commits_files_set
                repo_index.files_map.update({k: v for k, v in files_map.items() if k not in repo_index.files_map})
                EvaCode._save_repo_index(repo_index)

            self._repo_index_map[repo_index.uuid] = repo_index
            self._generate_repo_embeddings(repo_index)

        except Exception as e:
            logger.error(f"Failed to index: {repo_path} {e}")
            return False


class EvaCodeDaemon:
    DAEMON_IPC = f"ipc:///{tempfile.gettempdir()}/{APP_ID}"

    def __init__(self):
        self.exit_requested = False
        self.eva_code = EvaCode()

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        # self.socket.setsockopt(zmq.RCVTIMEO, 5000)
        self.socket.setsockopt(zmq.SNDTIMEO, 5000)

    def release(self):
        self.socket.close()
        self.context.term()

    def handle_request(self, request: CodeRequest):
        if request.request == CodeRequestType.STOP:
            self.send_reply(CodeReply(CodeStatus.OK))
            self.exit_requested = True
            return

        if request.request == CodeRequestType.ADD_REPO:
            repo_path = request.payload.get("repo_path")
            self.send_reply(CodeReply(CodeStatus.OK))
            self.eva_code.add_repo_to_index(repo_path)
        elif request.request == CodeRequestType.REMOVE_REPO:
            request.payload.get("repo_path")
            self.send_reply(CodeReply(CodeStatus.OK))
            # TODO Implement
        elif request.request == CodeRequestType.QUERY_REPOS:
            request.payload.get("repo_path")
            request.payload.get("query")
            self.send_reply(CodeReply(CodeStatus.OK))
            # TODO Implement
        else:
            return CodeReply.from_error("Error: Unknown CLI request", logger)

    def send_reply(self, reply: CodeReply):
        try:
            self.socket.send(msgspec.msgpack.encode(reply))
        except (BrokenPipeError, ConnectionResetError, OSError) as e:
            logger.error(f"Failed reply: {e}")

    def read_request_blocking(self) -> CodeRequest | None:
        try:
            return msgspec.msgpack.decode(self.socket.recv(), type=CodeRequest)
        except (BrokenPipeError, ConnectionResetError, OSError, msgspec.DecodeError) as e:
            self.send_reply(CodeReply.from_error(f"Invalid request: {e}", logger))
            return None

    def main_loop(self):
        logger.info("Begin daemon loop...")
        try:
            self.socket.bind(EvaCodeDaemon.DAEMON_IPC)
        except OSError as e:
            logger.error(f"Failed bind IPC {EvaCodeDaemon.DAEMON_IPC}: {e}")
            self.release()
            exit(1)

        while not self.exit_requested:
            try:
                request = self.read_request_blocking()
                if request:
                    self.handle_request(request)
            except Exception as e:
                logger.error(f"{e}")

        self.release()


def send_request(request_type: CodeRequestType, payload: dict | None = None) -> CodeReply | None:
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.RCVTIMEO, 5000)
    socket.setsockopt(zmq.SNDTIMEO, 5000)

    response = None
    try:
        socket.connect(EvaCodeDaemon.DAEMON_IPC)
        request = CodeRequest(request=request_type, payload=payload)
        socket.send(msgspec.msgpack.encode(request))
        response = msgspec.msgpack.decode(socket.recv(), type=CodeReply)
    except Exception as e:
        logger.error(f"Failed request: {e}")

    socket.close()
    context.term()
    return response


@click.group()
def cli():
    pass


@cli.command()
def start():
    daemon = EvaCodeDaemon()
    daemon.main_loop()


@cli.command()
def stop():
    send_request(CodeRequestType.STOP)


@cli.command()
@click.argument("repo-path")
def add_repo(repo_path: str):
    absolute_path = os.path.abspath(repo_path)
    send_request(CodeRequestType.ADD_REPO, {"repo_path": absolute_path})


@cli.command()
@click.argument("repo-path")
def remove_repo(repo_path: str):
    send_request(CodeRequestType.REMOVE_REPO, {"repo_path": repo_path})


if __name__ == "__main__":
    cli()
