"""Setup."""  # coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2023(18588220928@163.com), All Rights Reserved.
# ***
# ***    File Author: Dell, Mon 21 Aug 2023 03:06:09 PM CST
# ***
# ************************************************************************************/
#

from setuptools import setup

with open("README.md", "r") as file:
    long_description = file.read()

setup(
    name="Anime2Sketch",
    version="1.0.0",
    author="Dell Du",
    author_email="18588220928@163.com",
    description="Anime2Sketch Package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/delldu/Anime2Sketch.git",
    packages=["Anime2Sketch"],
    package_data={
        "Anime2Sketch": [
            "models/Anime2Sketch.pth",
        ]
    },
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "torch >= 1.9.0",
        "torchvision >= 0.10.0",
        "Pillow >= 7.2.0",
        "numpy >= 1.19.5",
        "todos >= 1.0.0",
    ],
)
