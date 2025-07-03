import time

file_path = "/root/paddlejob/workspace/env_run/output/liuyuanle/0_Origin_Models/ernie-4_5-21b-a3b-bf16-paddle/model-00008-of-00009.safetensors"

"""
# !test for load_file

# import paddle
# from safetensors.paddle import load_file
from safetensors.torch import load_file
# from safetensors.numpy import load_file
s = time.perf_counter()
tensors = load_file(file_path)
e = time.perf_counter()

print(f"tensor count : {len(tensors.keys())}\ntime: {(e-s)*1e3}ms")
for k,v in tensors.items():
    v = v + 100.0
    # print(k, v)
#     break
"""


# !test for safe_open
from safetensors import safe_open

import paddle

device = "cpu"  # or "cuda:0" for GPU
tensors = {}
def torch_test():
    s = time.perf_counter()
    with safe_open(file_path, framework="pt", device=device if device == "cpu" else "cuda") as f:
        for name in f.keys():
            tensors[name] = f.get_tensor(name)
    e = time.perf_counter()
    return (e-s)*1e3
def numpy_test():
    s = time.perf_counter()
    with safe_open(file_path, framework="np", device="cpu") as f:
        for name in f.keys():
            tensors[name] = f.get_tensor(name)
    e = time.perf_counter()
    return (e-s)*1e3
def paddlenlp_test():
    s = time.perf_counter()
    from paddle_safe_open import fast_safe_open
    with fast_safe_open(file_path, framework="np") as f:
        for k in f.keys():
            tensors[k] = paddle.to_tensor(f.get_tensor(k), place=device)
    e = time.perf_counter()
    return (e-s)*1e3
def paddle_test():
    s = time.perf_counter()
    with safe_open(file_path, framework="pp", device=device) as f:
        for name in f.keys():
            tensors[name] = f.get_tensor(name)
    e = time.perf_counter()
    return (e-s)*1e3

# t = torch_test()
# t = numpy_test()
# t = paddlenlp_test()
t = paddle_test()
for k, v in tensors.items():
    print(k, v)

print(f"tensor count : {len(tensors.keys())}\ntime(ms): {t}")

"""
file: /root/.paddlenlp/models/Qwen/Qwen2.5-7B-Instruct/model-00004-of-00004.safetensors
size: 3.4GB
tensors: 63

# !test record for with_open only cpu (ms)
numpy(MMap):                2471.426932141185
paddlenlp_get_numpy(MMap):  2390.2605809271336  # paddlenlp里的方法, python的MMap接口

pytorch(Storage):           1191.4707105606794
pytorch(MMap):              3627.9323790222406  # 对应paddle的读取方式
paddlenlp_get_tensor(MMap): 4321.236601099372   # 调用python里的MMap接口
paddle(MMap):               4344.542840495706   # 集成在rust里


# gpu
pytorch(Storage):           2277.809774503112
pytorch(MMap):              4415.908295661211
paddlenlp_get_tensor(MMap): 3139.2310429364443
padddle(MMap):              3118.960987776518

"""


"""
/root/paddlejob/workspace/env_run/output/liuyuanle/0_Origin_Models/ernie-4_5-21b-a3b-bf16-paddle
file: /root/.paddlenlp/models/Qwen/Qwen2.5-7B-Instruct/model-00004-of-00004.safetensors
size: 3.4GB
tensors: 63

torch load_file
cpu: 2.2564157843589783ms
gpu: 1236.2145856022835ms

paddle load_file
cpu: 4658.028410747647 ms
gpu: 3512.159211561084 ms


load_file:
numpy cpu: 2463.2687754929066ms
torch cpu: 6.569914519786835ms

"""

