#!/usr/bin/env bash

pip install -U twine
rm -rf dist/
python setup.py sdist
python setup.py bdist_wheel
twine upload dist/*
