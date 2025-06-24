import time
file_path = "/root/paddlejob/workspace/env_run/output/liuyuanle/0_Origin_Models/baidu/paddle_internal/ernie-x1-300b-a47b-bf16-paddle/model-00122-of-00122.safetensors"
"""
from safetensors import safe_open

tensors = {}
s = time.perf_counter()
with safe_open(file_path, framework="pt", device="cpu") as f:
   for key in f.keys():
       tensors[key] = f.get_tensor(key)
e = time.perf_counter()


"""

from safetensors.torch import load_file
# from safetensors.paddle import load_file
s = time.perf_counter()
tensors = load_file(file_path, device="cuda")
e = time.perf_counter()

# """

print(f"file size: 4.3GB\ntensor count: {len(tensors.keys())}\nload time: {(e - s)*1e3} ms")
for key, value in tensors.items():
    print(key, value)
    break
"""
test record

file: ernie-x1-300b-a47b-bf16-paddle/model-00122-of-00122.safetensors
size: 4.3GB
tnesor: 32个

torch with_open
cpu: 1244.6545772254467 ms
gpu: 4184.085264801979 ms

paddle_baseline with_open(MMap)
cpu: 9435.70539355278 ms
gpu: 7289.6928787231445 ms


torch load_file
cpu: 1.5714503824710846 ms
gpu: 3328.124813735485 ms

paddle_origin load_file
cpu: 7361.494410783052 ms
gpu: 5764.970671385527 ms


"""