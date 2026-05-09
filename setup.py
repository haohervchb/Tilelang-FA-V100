from setuptools import setup, find_packages

setup(
    name="tilelang-fa-v100",
    version="1.0.0",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=["tilelang>=0.1.9", "torch>=2.5.0"],
)
