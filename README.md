# Diffusion

This repo is intended for experimentation with Stable Diffusion pipelines. 

![alt text](example/out.png "Stable Diffusion Output")

*A cute and realistic kitten* 

## How to download model weights

`huggingface-cli download stabilityai/stable-video-diffusion-img2vid-xt`

## How to make a sandbox from a singularity simg file

First start an interactive job on JADE:

`srun -I --pty -t 0-10:00 --gres gpu:1 -p small bash`

Now run 
`singularity build --sandbox diffusion_sandbox/ diffusion.simg`

## Tips for running experiments

Scheduling a 10 minute job on Slurm, using singularity:
```bash
sbatch -t 10:0 -J stable_diffusion --gres gpu:1 -p devel --output ~/logs/%j.out --wrap
"singularity exec --nv ~/Documents/containers/diffusion_sandbox python stable_diffusion.py"
```

Using this bash command, you can output a text file into an array of prompts line-by-line:

`mapfile -t <example/prompts/list.txt prompts`

For zsh:

`prompts=("${(f)$(< example/prompts/list.txt)}")`

Now iterate over the prompts:

```bash
for prompt in "${prompts[@]}"
do
  python stable_diffusion.py prompt=$prompt
done
```
