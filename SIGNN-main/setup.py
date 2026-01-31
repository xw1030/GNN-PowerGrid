"""
Setup script for SIGNN: Symmetry Isomorphism Graph Neural Networks for Power Grid Analysis
Author: Charlotte Cambier van Nooten
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (
    (this_directory / "README.md").read_text()
    if (this_directory / "README.md").exists()
    else ""
)

# Read requirements from requirements.txt
requirements_path = this_directory / "requirements.txt"
if requirements_path.exists():
    with open(requirements_path) as f:
        requirements = [
            line.strip() for line in f if line.strip() and not line.startswith("#")
        ]
else:
    requirements = [
        "torch>=2.0.0",
        "pandas>=1.5.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
    ]

setup(
    name="signn",
    version="1.0.0",
    author="Charlotte Cambier van Nooten",
    description="Symmetry Isomorphism Graph Neural Networks for Power Grid n-1 Contingency Classification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/charlottecvn/SIGNN",  # Update with actual repository URL
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "full": [
            "torch-geometric>=2.3.0",
            "torch-scatter>=2.1.0",
            "torch-sparse>=0.6.15",
            "torch-cluster>=1.6.0",
            "torch-spline-conv>=1.2.1",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=5.0.0",
            "mypy>=0.991",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "nbsphinx>=0.8.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "signn-train=signn.cli:train",
            "signn-eval=signn.cli:evaluate",
            "signn-analyze=signn.cli:analyze",
        ],
    },
    include_package_data=True,
    package_data={
        "signn": ["data/*.csv", "configs/*.yaml"],
    },
    keywords=[
        "graph neural networks",
        "power systems",
        "contingency analysis",
        "n-1 security",
        "machine learning",
        "electrical engineering",
    ],
    project_urls={
        "Bug Reports": "https://github.com/charlottecvn/SIGNN/issues",
        "Source": "https://github.com/charlottecvn/SIGNN",
        "Documentation": "https://signn.readthedocs.io/",
    },
)
