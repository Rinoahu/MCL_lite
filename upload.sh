#!/bin/bash
#cd MCL_lite
git config --global user.email xiaohu@iastate.edu
git config --global user.name Rinoahu


git remote rm origin

git add -A .
git commit -m 'add a cluster comparison tool'
git remote add origin https://github.com/Rinoahu/MCL_lite

git pull origin master
git push origin master

git checkout master
