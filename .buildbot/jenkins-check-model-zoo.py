#!/usr/bin/env python3

import datetime
import git
import logging
import math
import os
import requests
import subprocess
import sys

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(lineno)03d] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Set parallel jobs based on both CPU count and memory size.
# Because using CPU count alone can result in out of memory
# and get Jenkins killed. For example, we may have 64 CPUs
# (128 threads) and only 32GB memory. So spawning off 128
# cc/c++ processes is going to quickly exhaust the memory.
#
# Algorithm: NPROC = min(2, # of CPUs) if memory < 8GB, otherwise
#            NPROC = min(memory / 8, # of CPUs)
MEMORY_IN_GB = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024.0**3)
NPROC = str(math.ceil(min(max(2, MEMORY_IN_GB / 8), os.cpu_count())))

DOCKER_DEV_IMAGE_WORKDIR = "/workdir"
ONNX_MLIR_SOURCE = "/workdir/onnx-mlir"
ONNX_MLIR_HOME = "/workdir/onnx-mlir/build/Debug"
RUN_ONNX_MODEL_PY = "RunONNXModel.py"
RUN_ONNX_MODELZOO_PY = "RunONNXModelZoo.py"

RENDERJSON_URL = "https://raw.githubusercontent.com/caldwell/renderjson/master/"
RENDERJSON_JS = "renderjson.js"

docker_daemon_socket = os.getenv("DOCKER_DAEMON_SOCKET")
docker_registry_host_name = os.getenv("DOCKER_REGISTRY_HOST_NAME")
docker_registry_user_name = os.getenv("DOCKER_REGISTRY_USER_NAME")
github_repo_name = os.getenv("GITHUB_REPO_NAME")
github_repo_name2 = os.getenv("GITHUB_REPO_NAME").replace("-", "_")
github_pr_baseref = os.getenv("GITHUB_PR_BASEREF")
github_pr_baseref2 = os.getenv("GITHUB_PR_BASEREF").lower()
github_pr_number = os.getenv("GITHUB_PR_NUMBER")
jenkins_home = os.getenv("JENKINS_HOME")
job_name = os.getenv("JOB_NAME")
workspace_dir = os.getenv("WORKSPACE")

modelzoo_reportdir = os.getenv("MODELZOO_REPORTDIR")
modelzoo_workdir = os.getenv("MODELZOO_WORKDIR")
modelzoo_html = os.getenv("MODELZOO_HTML")
modelzoo_stdout = os.getenv("MODELZOO_STDOUT")

docker_dev_image_name = (
    github_repo_name
    + "-dev"
    + ("." + github_pr_baseref2 if github_pr_baseref != "main" else "")
)
docker_dev_image_tag = github_pr_number.lower()
docker_dev_image_full = (
    (docker_registry_host_name + "/" if docker_registry_host_name else "")
    + (docker_registry_user_name + "/" if docker_registry_user_name else "")
    + docker_dev_image_name
    + ":"
    + docker_dev_image_tag
)

# History directory is just the job directory
workspace_historydir = os.path.join(jenkins_home, "jobs", job_name)
container_historydir = os.path.join(DOCKER_DEV_IMAGE_WORKDIR, job_name)

workspace_reportdir = os.path.join(workspace_dir, modelzoo_reportdir)
container_reportdir = os.path.join(DOCKER_DEV_IMAGE_WORKDIR, modelzoo_reportdir)

workspace_workdir = os.path.join(workspace_dir, modelzoo_workdir)
container_workdir = os.path.join(DOCKER_DEV_IMAGE_WORKDIR, modelzoo_workdir)

workspace_model_py = os.path.join(workspace_dir, "utils", RUN_ONNX_MODEL_PY)
container_model_py = os.path.join(DOCKER_DEV_IMAGE_WORKDIR, RUN_ONNX_MODEL_PY)

workspace_modelzoo_py = os.path.join(workspace_dir, "utils", RUN_ONNX_MODELZOO_PY)
container_modelzoo_py = os.path.join(DOCKER_DEV_IMAGE_WORKDIR, RUN_ONNX_MODELZOO_PY)


def urlretrieve(remote_url, local_file):
    req = requests.get(remote_url)
    with open(local_file, "wb") as f:
        f.write(req.content)


def main():
    repo = git.Repo(".")
    head_commit_message = repo.head.commit.message.split("\n", 1)[0]
    head_commit_author = "{} <{}>".format(
        repo.head.commit.author.name, repo.head.commit.author.email
    )
    head_commit_hash = repo.head.commit.hexsha
    head_commit_date = (
        datetime.datetime.utcfromtimestamp(repo.head.commit.committed_date).isoformat()
        + "Z"
    )

    cmd = [
        "docker",
        "run",
        "--rm",
        "-u",
        str(os.geteuid()) + ":" + str(os.getegid()),
        "-e",
        "ONNX_MLIR_HOME=" + ONNX_MLIR_HOME,
        "-e",
        "ONNX_MLIR_HEAD_COMMIT_MESSAGE=" + head_commit_message,
        "-e",
        "ONNX_MLIR_HEAD_COMMIT_AUTHOR=" + head_commit_author,
        "-e",
        "ONNX_MLIR_HEAD_COMMIT_HASH=" + head_commit_hash,
        "-e",
        "ONNX_MLIR_HEAD_COMMIT_DATE=" + head_commit_date,
        "-v",
        workspace_historydir + ":" + container_historydir,
        "-v",
        workspace_reportdir + ":" + container_reportdir,
        "-v",
        workspace_workdir + ":" + container_workdir,
        "-v",
        workspace_model_py + ":" + container_model_py,
        "-v",
        workspace_modelzoo_py + ":" + container_modelzoo_py,
        docker_dev_image_full,
        container_modelzoo_py,
        "-H",
        modelzoo_html,
        "-j",
        NPROC,
        "-l",
        "info",
        "-q",
        container_historydir,
        "-r",
        container_reportdir,
        "-w",
        container_workdir,
    ]

    # Write summary line to file for Jenkinsfile to pickup
    logging.info(" ".join(cmd))
    os.makedirs(workspace_workdir)
    os.makedirs(workspace_reportdir)
    with open(os.path.join(workspace_reportdir, modelzoo_stdout), "w") as f:
        try:
            proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.PIPE)

            # print messages from RunONNXModelZoo.py and RunONNXModel.py
            for line in proc.stderr:
                print(line.decode("utf-8"), file=sys.stderr, end="", flush=True)

            proc.wait()
        except:
            f.write("failed")

    # Download renderjson.js
    urlretrieve(
        RENDERJSON_URL + RENDERJSON_JS, os.path.join(workspace_reportdir, RENDERJSON_JS)
    )


if __name__ == "__main__":
    main()
