#!/bin/bash

FILES=/home/ed/Documents/repos/diffusers/data/statue_scene/*
PREFIX=/home/ed/Documents/repos/diffusers/data/

for f in $FILES
  do
  g=${f#$PREFIX}
  echo $g
done
