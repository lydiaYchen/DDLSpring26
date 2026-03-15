import torch

def get_device() -> torch.device:
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        device_str = "cuda"
    elif torch.mps.is_available():
        device_str = "mps"
    else:
        device_str = "cpu"

    return torch.device(device_str)
