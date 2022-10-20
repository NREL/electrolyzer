#!/usr/bin/env python

"""The setup script."""

from pathlib import Path
from setuptools import setup, find_packages

# Package meta-data.
NAME = "Electrolyzer"
DESCRIPTION = "A controls-oriented engineering electrolyzer model."
URL = "https://github.com/NREL/electrolyzer"
EMAIL = "christopher.bay@nrel.gov"
AUTHOR = "NREL National Wind Technology Center"
REQUIRES_PYTHON = ">=3.8.0"

# What packages are required for this module to be executed?
REQUIRED = [
    "numpy",
    "scipy",
    "matplotlib",
    "pandas",
]

# What packages are optional?
EXTRAS = {
    "docs": {"readthedocs-sphinx-ext", "Sphinx", "sphinxcontrib-napoleon"},
    "develop": {"pytest>=3"},
}

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

test_requirements = ['pytest>=3', ]

ROOT = Path(__file__).parent
with open(ROOT / "electrolyzer"/ "version") as version_file:
    VERSION = version_file.read().strip()

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=readme + '\n\n' + history,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(include=['electrolyzer', 'electrolyzer.*']),
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    license="Apache Software License 2.0",
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    test_suite='tests',
    tests_require=test_requirements,
    zip_safe=False,
)
