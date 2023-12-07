# Diffusion

This repo is intended for experimentation with Stable Diffusion pipelines. 

![alt text](misc/out.png "Stable Diffusion Output")

*A cute and realistic kitten* 

## Tips for running experiments

Using this bash command, you can output a text file into an array of prompts line-by-line:
`mapfile -t <misc/prompts/list.txt prompts`

For zsh:

`prompts=("${(f)$(< misc/prompts/list.txt)}")`

Now iterate over the prompts:

```bash
for prompt in "${prompts[@]}"
do
  python stable_diffusion.py prompt=$prompt
done
```
