from setuptools import setup, find_packages

setup(
    name="sprite-pipeline",
    version="0.1.0",
    author="Alexander Brodko",
    description="A pipeline for converting images into stylized game sprites with style transfer, enhancement, and sprite sheet generation.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/alexanderbrodko/sprite-pipeline",
    py_modules=["sp_group", "sp_pack"],
    install_requires=[
        "numpy",
        "opencv-python",
        "torch",
        "torchvision",
        "Pillow",
        "psd-tools",
        "rectpack",
        "modelscope",
        "basicsr",
        "realesrgan",
        "nst_vgg19",
        "gdown"
    ],

    entry_points={
        "console_scripts": [
            "sp_group=sp_group:main",
            "sp_pack=sp_pack:main"
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)