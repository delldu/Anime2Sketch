"""Image Recognize Anything Model Package."""  # coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2023(18588220928@163.com) All Rights Reserved.
# ***
# ***    File Author: Dell, Thu 13 Jul 2023 01:55:56 PM CST
# ***
# ************************************************************************************/
#

__version__ = "1.0.0"

import os
from tqdm import tqdm
from PIL import Image
import torch
import todos
import torch.nn.functional as F
from torchvision.transforms import ToTensor
from .anime2sketch import UnetGenerator

import pdb


def create_model():
    """
    Create model
    """

    model = UnetGenerator()
    device = todos.model.get_device()
    model = model.to(device)
    model.eval()
    print(f"Running model on {device} ...")

    return model, device


def get_model():
    """Load jit script model."""

    model, device = create_model()
    # print(model)

    # https://github.com/pytorch/pytorch/issues/52286
    torch._C._jit_set_profiling_executor(False)
    # C++ Reference
    # torch::jit::getProfilingMode() = false;
    # torch::jit::setTensorExprFuserEnabled(false);

    model = torch.jit.script(model)
    todos.data.mkdir("output")
    if not os.path.exists("output/Anime2Sketch.torch"):
        model.save("output/Anime2Sketch.torch")

    return model, device


def predict(input_files, output_dir):
    # Create directory to store result
    todos.data.mkdir(output_dir)

    # load model
    model, device = get_model()
    image_filenames = todos.data.load_files(input_files)

    # start predict
    progress_bar = tqdm(total=len(image_filenames))
    for filename in image_filenames:
        progress_bar.update(1)

        image = Image.open(filename).convert("RGB")
        input_tensor = ToTensor()(image).unsqueeze(0).to(device)
        B, C, H, W = input_tensor.size()
        input_backup = input_tensor.clone()
        input_tensor = F.interpolate(input_tensor, size=(512, 512), mode="bilinear", align_corners=True)

        with torch.no_grad():
            output_tensor = model(input_tensor)

        output_file = f"{output_dir}/{os.path.basename(filename)}"

        output_tensor = output_tensor.repeat(1, 3, 1, 1)
        output_tensor = F.interpolate(output_tensor, size=(H, W), mode="bilinear", align_corners=True)

        todos.data.save_tensor([input_backup, output_tensor], output_file)

    progress_bar.close()

    todos.model.reset_device()
