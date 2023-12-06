# Diffusion

This repo is intended for experimentation with Stable Diffusion pipelines.

## Tips for running experiments

Using this shell command, you can output a text file into an array of prompts line-by-line.
`prompts=("${(f)$(< conf/prompts/list.txt)}")`

