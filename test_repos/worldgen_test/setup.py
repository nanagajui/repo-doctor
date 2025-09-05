from setuptools import setup, find_packages

setup(
    name="worldgen-test",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0", 
        "transformers>=4.30.0",
        "flash-attn>=2.0.0",
        "numpy>=1.24.0",
        "opencv-python>=4.8.0",
        "pillow>=9.0.0",
    ],
    python_requires=">=3.8",
)
