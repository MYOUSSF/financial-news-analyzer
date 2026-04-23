"""
setup.py — Package configuration for Financial News Analyzer.

Install in editable mode for development:
    pip install -e .

Install for production:
    pip install .
"""
from setuptools import setup, find_packages
from pathlib import Path

# Read the long description from README
long_description = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")

# Read pinned dependencies from requirements.txt
requirements = [
    line.strip()
    for line in (Path(__file__).parent / "requirements.txt").read_text().splitlines()
    if line.strip() and not line.startswith("#")
]

setup(
    name="financial-news-analyzer",
    version="1.0.0",
    author="MYSSF",
    description="AI-powered financial news analysis and investment research assistant",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MYSSF/financial-news-analyzer",
    license="MIT",
    packages=find_packages(exclude=["tests*", "notebooks*", "docs*"]),
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=8.2.0",
            "pytest-asyncio>=0.23.6",
            "pytest-cov>=5.0.0",
            "pytest-mock>=3.14.0",
            "black>=24.4.2",
            "flake8>=7.0.0",
            "isort>=5.13.2",
            "mypy>=1.10.0",
        ],
        "docs": [
            "mkdocs>=1.6.0",
            "mkdocs-material>=9.5.20",
        ],
    },
    entry_points={
        "console_scripts": [
            "financial-analyzer=src.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Financial",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    package_data={
        "": ["config/*.yaml", "config/*.yml"],
    },
    include_package_data=True,
)
