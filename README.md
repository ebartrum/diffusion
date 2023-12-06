# Diffusion

This repo is intended for experimentation with Stable Diffusion pipelines. 

![alt text](example/out.png "Stable Diffusion Output")

*A cute and realistic kitten* 

## Tips for running experiments

Using this shell command, you can output a text file into an array of prompts line-by-line:

`prompts=("${(f)$(< conf/prompts/list.txt)}")`

