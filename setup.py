from distutils.core import setup

from setuptools import find_packages

CLASSIFIERS = [
    "Development Status :: 2 - Pre-Alpha",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python",
    "Topic :: Scientific/Engineering",
    "Topic :: Scientific/Engineering :: Mathematics",
    "Topic :: Scientific/Engineering :: Physics",
    "Operating System :: Microsoft :: Windows",
    "Operating System :: POSIX",
    "Operating System :: Unix",
    "Operating System :: MacOS",
    "Natural Language :: English",
]

setup(
    name="geoapps",
    version="0.1.4",
    packages=find_packages(),
    install_requires=["numpy", "h5py", "scipy", "geoh5py"],
    author="Mira Geoscience",
    author_email="dominiquef@mirageoscience.com",
    description="Open-sourced Applications in Geoscience",
    keywords="earth sciences",
    url="https://mirageoscience-apps.readthedocs.io/en/latest/",
    download_url="https://github.com/MiraGeoscience/mirageoscience-apps.git",
    classifiers=CLASSIFIERS,
    platforms=["Windows", "Linux", "Solaris", "Mac OS-X", "Unix"],
    license="MIT License",
    use_2to3=False,
)
