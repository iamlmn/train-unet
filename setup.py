#!/usr/bin/env python
# coding: utf-8

"""
    Setup package

"""
from os import path
from setuptools import setup, find_packages

# Get the long description from the README file
HERE = path.abspath(path.dirname(__file__))
with open(path.join(HERE, "README.md"), encoding="utf-8") as f:
    LONG_DESCRIPTION = f.read()

setup(
    name="train_unet",
    version="1.0",
    description="To use U-net architecture and train models for medical imaging at ease.",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author="Lakshmi Naarayanan",
    author_email="lakshminaarayananvs@rediffmail.com",
    packages=find_packages(),
    python_requires=">=3.6",
    install_requires=["numpy","keras==2.2.4","tensorflow==1.15.2","scipy","pandas","skimage","docopt","tqdm"],
)
