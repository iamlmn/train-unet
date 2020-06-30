#!/usr/bin/env python
# coding: utf-8

"""
    Setup package

"""
from os import path
from setuptools import setup, find_packages

# Get the long description from the README file
# HERE = path.abspath(path.dirname(__file__))
with open("README.md", encoding="utf-8") as f:
    LONG_DESCRIPTION = f.read()
DISTNAME = 'train-unet'

PACKAGE_NAME = 'train_unet'
setup(
    name="train_unet",
    version="0.0.1",
    description="To use U-net architecture and train models for medical imaging at ease.",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url = "https://github.com/iamlmn/train-unet",
    author="Lakshmi Naarayanan",
    author_email="lakshminaarayananvs@rediffmail.com",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    packages=find_packages(),
    include_package_data = False,
    python_requires=">=3.6",
    install_requires=["numpy","keras==2.2.4","tensorflow==1.13.1","scipy","pandas","skimage","docopt","tqdm"],
)
