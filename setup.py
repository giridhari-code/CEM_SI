"""
CEM Setup Script.
Allows installation via: pip install -e .
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="cem-model",
    version="0.1.0",
    author="Giridhari karmakar",
    author_email="giriisdev@gmail.com",
    description="Cognitive Emergence Machine (CEM) - Transformer-Free Looped Architecture",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/giridhari-code/CEM_SI",
    packages=find_packages(exclude=["tests*", "scripts*", "data*"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.1.0",
        "torchaudio>=2.1.0",
        "torchvision>=0.16.0",
        "numpy>=1.24.0",
        "Pillow>=10.0.0",
        "PyYAML>=6.0",
        "tqdm>=4.66.0",
        "safetensors>=0.4.0",
    ],
)