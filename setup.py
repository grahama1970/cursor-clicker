"""
Backward compatibility setup.py file.

This file is maintained for compatibility with older tooling that doesn't support pyproject.toml.
For modern Python packaging, refer to pyproject.toml.
"""

from setuptools import setup

if __name__ == "__main__":
    setup() 