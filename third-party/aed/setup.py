#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""A setuptools based setup module.
See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

# To use a consistent encoding
from codecs import open
from os import path

# Always prefer setuptools over distutils
from setuptools import find_packages, setup

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='aproximated_ged',

    version='0.1.0',

    description='Bunch of aproximated graph edit distance algorithms.',
    long_description=long_description,

    # The project's main homepage.
    url='https://github.com/priba/aproximated_ged',

    # Author details
    author='Pau Riba, Anjan Dutta',
    author_email='priba@cvc.uab.cat, adutta@cvc.uab.cat',

    # Choose your license
    license='MIT',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        'Development Status :: 3 - Alpha',

        'Intended Audience :: Science/Research',

        'License :: OSI Approved :: MIT License',

        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],

    keywords='ged graph',

    packages=find_packages(exclude=['contrib', 'docs', 'tests', 'tasks', 'scripts', 'bak', 'main.py']),

    install_requires=[], #'networkx==1.11', 'numpy==1.13.1', 'scikit-learn==0.18.1'],

    extras_require={
        'dev': ['check-manifest'],
        'test': ['coverage'],
    },

    python_requires='>=3',
)
