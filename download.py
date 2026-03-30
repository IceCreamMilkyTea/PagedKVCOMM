import os
from huggingface_hub import snapshot_download

try:
    snapshot_download(
        repo_id="meta-llama/Llama-3.1-8B-Instruct",
        local_dir="/home/users/hz314/models/Llama-3.1-8B-Instruct/",
        local_dir_use_symlinks=False,
        resume_download=True,
        token=True
    )
    print("下载成功！")
except Exception as e:
    print(f"下载失败，错误信息: {e}")