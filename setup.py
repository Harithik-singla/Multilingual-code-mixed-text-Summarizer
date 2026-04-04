"""
Package installer. Defines the `codemix_summarizer` package and CLI entry points
for `train` and `summarize` commands.
"""


from setuptools import setup, find_packages
from typing import List

# find_packages() automatically discovers all packages and sub-packages, i.e., folders with __init__.py files

def get_requirements() -> List[str]:
    """Read requirements.txt and return dependencies (skip editable flag)."""
    requirements: List[str] = []
    try:
        with open("requirements.txt", "r", encoding="utf-8") as file:
            for line in file:
                # Drop inline comments and trim whitespace
                cleaned = line.split("#", 1)[0].strip()
                if cleaned and cleaned != "-e .":
                    requirements.append(cleaned)
    except FileNotFoundError:
        print("requirements.txt file not found.")

    return requirements


setup(
    name="codemix_summarizer",
    version="0.1.0",
    author="Raunaq Mittal",
    author_email="raunaqmittal2004@gmail.com",
    description="Abstractive summarization for code-mixed multilingual text (Indic languages + English)",
    long_description="A deep learning system for summarizing code-mixed text across Indian languages and English using transformer models.",
    packages=find_packages(),
    install_requires=get_requirements(),
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "codemix-summarizer=src.pipelines.inference_pipeline:main",
        ]
    }
)