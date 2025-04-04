# Diffusion

This repo is intended for experimentation with Stable Diffusion pipelines. 

![alt text](misc/out.png "Stable Diffusion Output")

*A cute and realistic kitten* 

## How to download model weights

`huggingface-cli download stabilityai/stable-video-diffusion-img2vid-xt`

## How to make a sandbox from a singularity simg file
`srun -t 0-30:00 --gres gpu:1 -p small singularity build --sandbox diffusion_sandbox/ diffusion.simg`

## Tips for running experiments

Scheduling a 10 minute job on Slurm, using singularity:
```bash
sbatch -t 10:0 -J stable_diffusion --gres gpu:1 -p devel --output ~/logs/%j.out --wrap
"singularity exec --nv ~/Documents/containers/diffusion_sandbox python stable_diffusion.py"
```

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
