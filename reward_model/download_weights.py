from huggingface_hub import snapshot_download
import os

save_root_path = "./"
model_urls = ["tloen/alpaca-lora-7b"]
save_paths = ["/home/jiashuo/codes/Muffin/reward_model/Llama/tloen/alpaca-lora-7b"]
local_dir_use_symlinks = False
token = ""

for model_url, save_path in zip(model_urls, save_paths):
    save_path = os.path.join(save_root_path, save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    SNAPSHOT_PATH = snapshot_download(model_url, cache_dir=save_path, etag_timeout=300, max_workers=4,
                                      local_dir_use_symlinks=local_dir_use_symlinks, token=token)
    print(f"Downloaded files are located in: {SNAPSHOT_PATH}")
