#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VMAS 覆盖任务可视化：导出 MP4
"""

import argparse
import os

import imageio.v2 as imageio
import torch
from torch import multiprocessing
from torchrl.envs import RewardSum, TransformedEnv
from torchrl.envs.libs.vmas import VmasEnv

from scenario_coverage import Scenario
from train_coverage_vmas_mappo import build_policy


def render_frame(env):
    frame = None
    try:
        frame = env.render(mode="rgb_array")
    except TypeError:
        frame = env.render()
    return frame


def main():
    parser = argparse.ArgumentParser(description="VMAS 覆盖任务可视化 (MP4)")
    parser.add_argument("--model_path", default="models/coverage_vmas_policy_best.pth")
    parser.add_argument("--out_dir", default="videos")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--width", type=float, default=1.0)
    parser.add_argument("--height", type=float, default=1.0)
    parser.add_argument("--grid_w", type=int, default=10)
    parser.add_argument("--grid_h", type=int, default=10)
    parser.add_argument("--revisit_limit", type=int, default=10)
    parser.add_argument("--randomize_area", action="store_true")
    parser.add_argument("--width_min", type=float, default=0.8)
    parser.add_argument("--width_max", type=float, default=1.2)
    parser.add_argument("--height_min", type=float, default=0.8)
    parser.add_argument("--height_max", type=float, default=1.2)
    parser.add_argument("--coverage_margin", type=float, default=0.9)
    args = parser.parse_args()

    is_fork = multiprocessing.get_start_method() == "fork"
    device = torch.device(0) if torch.cuda.is_available() and not is_fork else torch.device("cpu")
    vmas_device = device

    scenario = Scenario(
        width=args.width,
        height=args.height,
        grid_w=args.grid_w,
        grid_h=args.grid_h,
        revisit_limit=args.revisit_limit,
        max_steps=args.max_steps,
        randomize_area=args.randomize_area,
        width_range=(args.width_min, args.width_max),
        height_range=(args.height_min, args.height_max),
        coverage_margin=args.coverage_margin,
    )
    env = VmasEnv(
        scenario=scenario,
        num_envs=1,
        continuous_actions=True,
        max_steps=args.max_steps,
        device=vmas_device,
    )
    env = TransformedEnv(
        env,
        RewardSum(in_keys=[env.reward_key], out_keys=[("agents", "episode_reward")]),
    )

    policy = build_policy(env, device)
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"未找到模型: {args.model_path}")
    policy.load_state_dict(torch.load(args.model_path, map_location=device))
    policy.eval()

    os.makedirs(args.out_dir, exist_ok=True)

    for ep in range(args.episodes):
        frames = []
        td = env.reset(seed=args.seed + ep)
        frame = render_frame(env)
        if frame is not None:
            frames.append(frame)

        with torch.no_grad():
            for _ in range(args.max_steps):
                td = policy(td)
                td = env.step(td)
                frame = render_frame(env)
                if frame is not None:
                    frames.append(frame)
                td_next = td.get("next")
                if td_next is not None and td_next.get("done").any().item():
                    break
                td = td_next

        out_path = os.path.join(args.out_dir, f"coverage_vmas_episode_{ep + 1}.mp4")
        if frames:
            imageio.mimsave(out_path, frames, fps=args.fps)
            print(f"保存视频: {out_path} (frames={len(frames)})")

    env.close()


if __name__ == "__main__":
    main()
