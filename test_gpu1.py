import torch
import time

print(torch.__version__)        # 返回pytorch的版本
print(torch.version.cuda)
print(torch.cuda.is_available())        # 当CUDA可用时返回True
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))
print(torch.cuda.current_device())
print(torch.cuda.get_device_properties(0))
