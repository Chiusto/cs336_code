import torch
import numpy as np
import numpy.typing as npt  # ← 添加这一行

def data_loading(
    dataset: npt.NDArray, 
    batch_size: int, 
    context_length: int, 
    device: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
    # np.random.randint(low, high, size) 会生成 size 个随机整数，每个整数都在 [low, high) 范围内
    starts = np.random.randint(0,len(dataset)-context_length,size=batch_size)
    # np.stack() 会沿着新维度将这些序列堆叠起来
    inputs = np.stack([
        dataset[i:i + context_length]
        for i in starts
    ])
    targets = np.stack([
        dataset[i+1:i + context_length+1]
        for i in starts
    ])
    inputs = torch.tensor(inputs,dtype=torch.long,device=device)
    targets = torch.tensor(targets,dtype=torch.long,device=device)
    return inputs,targets