#!/bin/bash

FILES=/home/ed/Documents/repos/diffusers/data/statue_scene/*
PREFIX=/home/ed/Documents/repos/diffusers/

for f in $FILES
  do
  g=${f#$PREFIX}
  echo $g
  python inversion_heatmap.py $g
done
