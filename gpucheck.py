import torch

# Check if CUDA is available
print("CUDA Available:", torch.cuda.is_available())

# Check the number of GPUs
print("GPU Count:", torch.cuda.device_count())

# Check the name of the GPU
if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))

# Check the current device
print("Current CUDA Device:", torch.cuda.current_device())
