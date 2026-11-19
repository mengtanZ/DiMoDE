"""Runtime setup shared by training and evaluation."""
from pathlib import Path
import torch


def get_device(opt):
    if opt.no_cuda:
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Check the NVIDIA driver/GPU, or use --no_cuda for CPU execution.")
    return torch.device(f"cuda:{opt.device_id}")


def output_directory(opt):
    path = Path(opt.eval_out_dir or Path("eval") / opt.model_name)
    path.mkdir(parents=True, exist_ok=True)
    return path
