import os
import sys
from setuptools import setup, find_packages

install_requires = [
    "scprep>=0.10.0",
    "phate>=0.4.0",
    "scipy>=1.2.0",
    "scikit-learn",
    "graphtools>=1.0.0",
    "joblib",
]

test_requires = ["nose", "nose2", "coverage", "coveralls"]

doc_requires = [
    "sphinx",
    "sphinxcontrib-napoleon",
]

scripts_requires = ["umap-learn", "networkx", "rpy2"]

if sys.version_info[:2] < (3, 5):
    raise RuntimeError("Python version >=3.5 required.")

version_py = os.path.join(os.path.dirname(__file__), "demap", "version.py")
version = open(version_py).read().strip().split("=")[-1].replace('"', "").strip()

readme = open("README.rst").read()

setup(
    name="demap",
    version=version,
    description="demap",
    author="Scott Gigante, Krishnaswamy Lab, Yale University",
    author_email="krishnaswamylab@gmail.com",
    packages=find_packages(),
    license="GNU General Public License Version 3",
    install_requires=install_requires,
    extras_require={
        "test": test_requires,
        "doc": doc_requires,
        "scripts": scripts_requires,
    },
    test_suite="nose2.collector.collector",
    long_description=readme,
    url="https://github.com/KrishnaswamyLab/demap",
    download_url="https://github.com/KrishnaswamyLab/demap/archive/v{}.tar.gz".format(
        version
    ),
    keywords=["big-data", "computational-biology", "visualization",],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "Framework :: Jupyter",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
)
