# Train oasis

## Install

```bash
conda create -n oasis python=3.10
pip install -r requirements.txt
```
Install deepspeed with
```bash
DS_BUILD_CPU_ADAM=1 pip install deepspeed
```

## Data setup

Flappy Bird dataset on huggingface:

https://huggingface.co/quantumiracle-git/flappy_bird_video_action

untar the dataset to `data/flappy_bird`. 
The folder structure should be like this:
```
data/
├── flappy_bird/
│   ├── collected/
│       ├── xxx.mp4
│       └── xxx.pt
```

```bash
tar -xzvf data/flappy_bird.tar.gz -C data/
```


## Run

1. Set `CUDA_VISIBLE_DEVICES` in `train_oasis/main.py`.
2. Set `config_path` and `config_name` in `train_oasis/main.py`
3. Download vae ckpt (`vit-l-20.safetensors` from [here](https://huggingface.co/Etched/oasis-500m/tree/main)), put it under dir `vae/`, and make sure the vae_ckpt in `config/algorithm/df_video.yaml` to the path of the vae ckpt.
4. Set config in `config/dataset/flappy_bird_fast.yaml`: save_dir should be the path to the flappy bird dataset
5. Set some config in `config/flappy_bird.yaml`: 

    - wandb
    - experiment.training.batch_size

### Local training
```bash
python train_oasis/main.py
```

### Slurm training on server
This will copy current version of the code and config to a new folder, and submit a job to slurm, which is specified in `train.sh`.

You need to comment out the `os.environ["CUDA_VISIBLE_DEVICES"] = "0"` in `main.py` first, because the number of GPUs is specified in `train.sh`.

Run the following command to submit a job to slurm:
```bash
bash launch.sh
```

### To run the code

The inference process is autoregressive diffusion, similar to Diffusion Forcing.
Specify the gpu id for inference in `train_oasis/inference/bird.py`.

We provide a pre-trained ckpt for flappy bird.
Download the ckpt from [here](https://huggingface.co/cilae/flappy_bird_dit/tree/main), and put it under `outputs/flappy_bird_dit/window_size=30-step=36000.bin`.

Specify the ckpt path `--oasis-ckpt` as the provided ckpt path or your own path in `train_oasis/inference/bird.py` and run the following command:

```bash
python train_oasis/inference/bird.py
```

