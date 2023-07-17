from setuptools import setup, find_packages

VERSION = '0.0.1'
DESCRIPTION = 'Space–time slice plotting package for Python'
LONG_DESCRIPTION = 'Space–time slice plotting package for Python'

setup(
    name="sunslicepy",
    version=VERSION,
    author="Trestan Simon",
    author_email="trestansimon@gmail.com",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=[],

    keywords=['python', 'solar physics'],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Physics",
    ]
)
