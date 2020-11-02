#!/bin/sh

rm -f data/*png
rm -f data/*dat
rm -f data/*vts
find . -type d -name .ipynb_checkpoints | xargs rm -rf
find . -type d -name __pycache__ | xargs rm -rf



jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace *.ipynb
jupyter nbconvert --to python *.ipynb

cd src/
jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace *.ipynb
jupyter nbconvert --to python *.ipynb


