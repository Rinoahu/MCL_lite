#!/bin/bash
#cd MCL_lite
rm -rf __pycache__
git config --global user.email xiaohu@iastate.edu
git config --global user.name Rinoahu


git remote rm origin

git add -A .
git commit -m 'reduce disk usage by reducing the matrix size'
git remote add origin https://github.com/Rinoahu/MCL_lite

git pull origin master
git push origin master

git checkout master
