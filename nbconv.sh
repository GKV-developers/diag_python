#!/bin/sh

find ./data/ -name *.png  | xargs rm -f
find ./data/ -name *.dat  | xargs rm -f
find ./data/ -name *.vti  | xargs rm -f
find ./data/ -name *.vts  | xargs rm -f
find ./data/ -name *.pvts | xargs rm -f
find ./data/ -name *.xmf  | xargs rm -f
find ./data/ -name *.bin  | xargs rm -f
find . -type d -name .ipynb_checkpoints | xargs rm -rf
find . -type d -name __pycache__ | xargs rm -rf



jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace *.ipynb
jupyter nbconvert --to python *.ipynb

cd src/
jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace *.ipynb
jupyter nbconvert --to python *.ipynb


