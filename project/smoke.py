# coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2023(18588220928@163.com), All Rights Reserved.
# ***
# ***    File Author: Dell, Mon 21 Aug 2023 03:06:09 PM CST
#
# ************************************************************************************/
#
import pdb
import os
import torch
import Anime2Sketch

from tqdm import tqdm

if __name__ == "__main__":
    model, device = Anime2Sketch.get_model()
    # model = torch.jit.script(model)
    # print(model)

    B, C, H, W = 1, 3, 512, 512

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    ) as p:
        for ii in range(10):
            x = torch.randn(B, C, H, W)
            with torch.no_grad():
                y = model(x.to(device))
            torch.cuda.synchronize()
        p.step()

    print(p.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))
    os.system("nvidia-smi | grep python")
