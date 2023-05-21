from huggingface_hub import hf_hub_download
import joblib

REPO_ID = "shi-labs/versatile-diffusion"
# FILENAME = "pretrained_pth/kl-f8.pth"
FILENAME = "pretrained_pth/vd-four-flow-v1-0-fp16.pth"


address = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)#, local_dir="/root/pretrained", subfolder="pretrained_pth")
print(address)