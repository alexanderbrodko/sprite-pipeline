from setuptools import setup, find_packages

def read_requirements():
    with open('requirements.txt', encoding="utf-8") as req_file:
        return [line.strip() for line in req_file if not line.startswith('#') and line.strip()]

setup(
    name="sprite-pipeline",
    version="0.3.1",
    author="Alexander Brodko",
    description="A pipeline for converting images into stylized game sprites and sprite sheet generation.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/alexanderbrodko/sprite-pipeline",
    py_modules=["sp_group", "sp_pack", "sp_fix"],
    install_requires=read_requirements(),
    package_data={
        'sp_group': ['models/*'],
    },
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "sp_group=sp_group:main",
            "sp_pack=sp_pack:main",
            "sp_fix=sp_fix:main"
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
