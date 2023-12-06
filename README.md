# Diffusion

This repo is intended for experimentation with Stable Diffusion pipelines. 

![alt text](example/out.png "Stable Diffusion Output")

*A cute and realistic kitten* 

## Tips for running experiments

Using this bash command, you can output a text file into an array of prompts line-by-line:
`mapfile -t <example/prompts/list.txt prompts`

For zsh:

`prompts=("${(f)$(< example/prompts/list.txt)}")`

Now iterate over the prompts:

for prompt in "${prompts[@]}"
do
  python stable_diffusion.py prompt=$prompt
done
