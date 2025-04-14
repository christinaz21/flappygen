"""
References:
    - Diffusion Forcing: https://github.com/buoyancy99/diffusion-forcing
"""
import sys
import os
dir_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(dir_path)

import torch
from train_oasis.model.dit import DiT_models
from diffusers.models import AutoencoderKL
from torchvision.io import read_video, write_video
from train_oasis.utils import sigmoid_beta_schedule
from tqdm import tqdm
from einops import rearrange
from torch import autocast
from safetensors.torch import load_model
import argparse
from pprint import pprint
import os
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
from pathlib import Path
from train_oasis.utils import parse_flappy_bird_action
import numpy as np
import pickle
from torchvision import transforms

assert torch.cuda.is_available()
device = torch.device("cuda:0") # TODO: change to your own GPU id for inference

def main(args):
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    if args.dtype == "float16":
        torch.set_default_dtype(torch.float16)
        dtype = torch.float16
    elif args.dtype == "float32":
        torch.set_default_dtype(torch.float32)
        dtype = torch.float32
    else:
        raise ValueError(f"unsupported dtype: {args.dtype}")

    # load DiT checkpoint
    model = DiT_models[args.model_name]()
    print(f"loading checkpoint from {os.path.abspath(args.oasis_ckpt)}...")
    if os.path.isdir(args.oasis_ckpt):
        ckpt = get_fp32_state_dict_from_zero_checkpoint(args.oasis_ckpt)
        state_dict = {}
        for key, value in ckpt.items():
            if key.startswith("diffusion_model."):
                state_dict[key[16:]] = value
        model.load_state_dict(state_dict, strict=True)
    elif args.oasis_ckpt.endswith(".pt"):
        ckpt = torch.load(args.oasis_ckpt, weights_only=True)
        model.load_state_dict(ckpt, strict=False)
    elif args.oasis_ckpt.endswith(".safetensors"):
        load_model(model, args.oasis_ckpt)
    elif args.oasis_ckpt.endswith(".bin"):
        ckpt = torch.load(args.oasis_ckpt)
        state_dict = {}
        for key, value in ckpt.items():
            if key.startswith("diffusion_model."):
                state_dict[key[16:]] = value
        model.load_state_dict(state_dict, strict=True)
    else:
        raise ValueError(f"unsupported checkpoint format: {args.oasis_ckpt}")
    model.max_frames = args.max_frames
    model = model.to(device).eval()

    # load VAE checkpoint
    vae = AutoencoderKL.from_pretrained("models/sd-vae-ft-ema")
    vae = vae.to(device, dtype=dtype).eval()

    # sampling params
    n_prompt_frames = args.n_prompt_frames
    total_frames = args.num_frames
    max_noise_level = 1000
    vae_batch_size = 32
    ddim_noise_steps = args.ddim_steps
    noise_range = torch.linspace(-1, max_noise_level - 1, ddim_noise_steps + 1, dtype=torch.long)
    print(noise_range)
    noise_range = torch.where(noise_range >= max_noise_level, max_noise_level - 1, noise_range)
    assert (noise_range < max_noise_level).all(), f"noise range out of bound: {noise_range}"
    noise_abs_max = 6 # open oasis use 20
    stabilization_level = 15

    # get prompt image/video
    action_path = Path(args.actions_path)
    video_path = Path("colored/videos/flappy_bird_night.mp4")
    assert video_path.exists(), f"File not found: {video_path}"

    prompt = read_video(action_path.with_suffix(".mp4"), pts_unit="sec")[0]
    prompt = read_video(video_path, pts_unit="sec")[0]
    print("prompt: ", prompt.shape)
    #print("prompt2: ", prompt2.shape)
    if args.video_offset is not None:
        prompt = prompt[args.video_offset:]
    prompt = prompt[:n_prompt_frames]
    prompt = prompt.float() / 255.0
    prompt = rearrange(prompt, "t h w c -> t c h w")
    if args.reduce_reso_rate > 1:
        transform = transforms.Resize(
            (prompt.shape[2] // args.reduce_reso_rate, prompt.shape[3] // args.reduce_reso_rate), antialias=True
        )
        prompt = transform(prompt)
    x = rearrange(prompt, "t c h w -> 1 t c h w")
    x = x.to(dtype=dtype)

    print(x.shape)
    # get input action stream
    with open(args.actions_path, "rb") as f:
        data = pickle.load(f)
    actions = [np.array([0,0])] + [parse_flappy_bird_action(d.item()) for d in data]
    if args.video_offset is not None:
        actions = actions[args.video_offset:]
    actions = actions[:total_frames]
    actions = np.array(actions)
    actions = torch.from_numpy(actions).unsqueeze(0).to(x.dtype) # (1, T, 2)
    assert actions.shape[1] == total_frames, f"{actions.shape[1]} != {total_frames}"
    print(actions.shape)
    # sampling inputs
    B = x.shape[0]
    H, W = x.shape[-2:]
    # x = torch.randn((B, 1, *x.shape[-3:]), device=device) # (B, 1, C, H, W)
    # x = torch.clamp(x, -noise_abs_max, +noise_abs_max)
    x = x.to(device)
    actions = actions.to(device)

    # vae encoding
    x = rearrange(x, "b t c h w -> (b t) c h w")
    with torch.no_grad():
        x = x * 2 - 1
        x = vae.encode(x).latent_dist.sample() * vae.config.scaling_factor
    x = rearrange(x, "(b t) ... -> b t ...", b=B, t=n_prompt_frames)

    # get alphas
    betas = sigmoid_beta_schedule(max_noise_level).float().to(device, dtype=dtype)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod = rearrange(alphas_cumprod, "T -> T 1 1 1")

    # sampling loop
    if args.inference_method == "single":
        for i in tqdm(range(n_prompt_frames, total_frames, args.chunk_size)):
            chunk = torch.randn((B, args.chunk_size, *x.shape[-3:]), device=device) # (B, 1, C, H, W)
            chunk = torch.clamp(chunk, -noise_abs_max, +noise_abs_max)
            chunk = chunk.to(dtype=dtype)
            curr_len = x.shape[1]
            x = torch.cat([x, chunk], dim=1)
            start_frame = max(0, i + args.chunk_size - model.max_frames)

            for noise_idx in reversed(range(1, ddim_noise_steps + 1)):
                # set up noise values
                t_ctx = torch.full((B, curr_len), stabilization_level - 1, dtype=torch.long, device=device)
                t = torch.full((B, args.chunk_size), noise_range[noise_idx], dtype=torch.long, device=device)
                t_next = torch.full((B, args.chunk_size), noise_range[noise_idx - 1], dtype=torch.long, device=device)
                t_next = torch.where(t_next < 0, t, t_next)
                t = torch.cat([t_ctx, t], dim=1)
                t_next = torch.cat([t_ctx, t_next], dim=1)

                # sliding window
                max_index = min(start_frame + model.max_frames, total_frames)
                x_curr = x.clone()
                x_curr = x_curr[:, start_frame:max_index]
                t = t[:, start_frame:max_index]
                t_next = t_next[:, start_frame:max_index]
                max_action_index = min(start_frame + x_curr.shape[1], total_frames)
                actions_curr = actions[:, start_frame : max_action_index]

                # get model predictions
                with torch.no_grad():
                    with autocast("cuda", dtype=dtype):
                        v = model(x_curr, t, actions_curr)
                
                if args.predict_v:
                    x_start = alphas_cumprod[t].sqrt() * x_curr - (1 - alphas_cumprod[t]).sqrt() * v
                else:
                    x_start = v
                x_noise = ((1 / alphas_cumprod[t]).sqrt() * x_curr - x_start) / (1 / alphas_cumprod[t] - 1).sqrt()

                # get frame prediction
                alpha_next = alphas_cumprod[t_next]
                alpha_next[:, :-1] = torch.ones_like(alpha_next[:, :-1])
                if noise_idx == 1:
                    alpha_next[:, -1:] = torch.ones_like(alpha_next[:, -1:])
                x_pred = alpha_next.sqrt() * x_start + x_noise * (1 - alpha_next).sqrt()
                x[:, -args.chunk_size:] = x_pred[:, -args.chunk_size:]
            x = x[:, :total_frames] # with chunk_size>1, x may be longer than total_frames
    else:
        raise ValueError(f"unsupported inference method: {args.inference_method}")

    # vae decoding
    x = rearrange(x, "b t ... -> (b t) ...")
    with torch.no_grad():
        all_frames = []
        for idx in tqdm(range(0, x.shape[0], vae_batch_size), desc="vae decoding frames"):
            x_clip = x[idx:idx + vae_batch_size]
            x_clip = vae.decode(x_clip / vae.config.scaling_factor).sample
            x_clip = (x_clip + 1) / 2
            all_frames.append(x_clip)
        x = torch.cat(all_frames, dim=0)
        x = rearrange(x, "(b t) c h w -> b t h w c", t=total_frames)

    # save video
    x = torch.clamp(x, 0, 1)
    x = (x * 255).byte()
    print("Output video shape:", x.shape)
    write_video(args.output_path, x[0].cpu(), fps=args.fps)
    print(f"generation saved to {args.output_path}.")


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    print("parsing args")

    parse.add_argument(
        "--oasis-ckpt",
        type=str,
        help="Path to Oasis DiT checkpoint.",
        default="outputs/flappy_bird_dit/window_size=30-step=36000.bin",
    )
    parse.add_argument(
        "--inference-method",
        type=str,
        help="Inference method to use.",
        choices=["single"],
        default="single",
    )
    parse.add_argument(
        "--model-name",
        type=str,
        help="Model name",
        default="flappy_bird_dit_half",
    )
    parse.add_argument(
        "--max-frames",
        type=int,
        help="Max frames",
        default=30,
    )
    parse.add_argument(
        "--predict_v",
        action="store_true",
        help="Whether the model use predict_v.",
        default=False,
    )
    parse.add_argument(
        "--num-frames",
        type=int,
        help="How many frames should the output be?",
        default=240,
    )
    parse.add_argument(
        "--actions-path",
        type=str,
        help="File to load actions",
        default="data/flappy_bird/collected/fb476858-00e7-4494-881f-3d370e667ecc.pkl",
    )
    parse.add_argument(
        "--chunk-size",
        type=int,
        help="Inference chunk size.",
        default=3,
    )
    parse.add_argument(
        "--video-offset",
        type=int,
        help="If loading prompt from video, index of frame to start reading from.",
        default=None,
    )
    parse.add_argument(
        "--reduce-reso-rate",
        type=int,
        help="Reduce resolution rate.",
        default=2,
    )
    parse.add_argument(
        "--n-prompt-frames",
        type=int,
        help="If the prompt is a video, how many frames to condition on.",
        default=1,
    )
    parse.add_argument(
        "--output-path",
        type=str,
        help="Path where generated video should be saved.",
        default="outputs/video/bird-night.mp4",
    )
    parse.add_argument(
        "--fps",
        type=int,
        help="What framerate should be used to save the output?",
        default=30,
    )
    parse.add_argument(
        "--dtype",
        type=str,
        help="What dtype should be used for the model?",
        default="float16",
    )
    parse.add_argument("--ddim-steps", type=int, help="How many DDIM steps?", default=10)

    args = parse.parse_args()
    print("inference args:")
    pprint(vars(args))
    main(args)
