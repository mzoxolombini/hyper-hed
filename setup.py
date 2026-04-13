"""Setup script for ILS-HED package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="ils-hed",
    version="1.0.0",
    author="Mzoxolo Mbini",
    author_email="u16350244@tuks.co.za",
    description="ILS-HED: Enhancing Holistically-Nested Edge Detection with Iterative Local Search Hyper-heuristics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mzoxolombini/ils-hed",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.12.0",
        "torchvision>=0.13.0",
        "opencv-python>=4.6.0",
        "scikit-image>=0.19.0",
        "numpy>=1.23.0",
        "scipy>=1.9.0",
        "scikit-learn>=1.1.0",
        "Pillow>=9.0.0",
        "tqdm>=4.64.0",
        "pyyaml>=6.0",
    ],
    entry_points={
        "console_scripts": [
            "ils-hed=ils_hed.train:main",
        ],
    },
)