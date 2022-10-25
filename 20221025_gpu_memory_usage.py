import torch
import numpy as np

import time
import gc

import GPUtil

t = np.ones(10000000).reshape(10000, 1000)

tt = torch.Tensor(t)

tt.to("cuda:0")
print("tt to device")
print(".")
print(".")
time.sleep(10)

# この時点ではGPU Memory Usageは減らない --
del tt
print("tt deleted")
print(".")
print(".")
time.sleep(10)

# これで減るはず?
torch.cuda.empty_cache()
print("torch.cuda.empty_cache()")
print(".")
print(".")
time.sleep(30)
