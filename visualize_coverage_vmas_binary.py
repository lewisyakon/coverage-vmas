#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VMAS 覆盖任务可视化（阈值跳变版）：10分钟内绿色，否则红色
"""

import argparse
import os

import imageio.v2 as imageio
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torch import multiprocessing
from torchrl.envs import RewardSum, TransformedEnv
from torchrl.envs.libs.vmas import VmasEnv

from scenario_coverage import Scenario
from train_coverage_vmas_mappo import build_policy


def set_fixed_camera(env, padding_ratio=0.02):
    """Fix camera by locking scenario zoom/origin used by VMAS render()."""
    base_env = env.base_env
    scenario = base_env.scenario
    device = base_env.device
    half_w = scenario.width * 0.5 * (1.0 + padding_ratio)
    half_h = scenario.height * 0.5 * (1.0 + padding_ratio)
    # VMAS shared_viewer branch final cam_range roughly scales as zoom^2.
    # Set zoom so that final visible half-range is at least the scenario bounds.
    target_half = max(float(half_w), float(half_h))
    scenario.viewer_zoom = max(1e-6, target_half ** 0.5)
    scenario.render_origin = torch.zeros(2, device=device)


def render_frame(env):
    # VMAS render() recomputes bounds every frame; lock zoom/origin beforehand.
    set_fixed_camera(env)
    try:
        frame = env.render(mode="rgb_array")
    except TypeError:
        frame = env.render()
    return frame


def annotate_frame(frame, max_gap, fresh_ratio, cell_w, cell_h):
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    lines = [
        f"Max gap: {max_gap:.1f} min",
        f"Fresh ratio: {fresh_ratio:.1%}",
        f"Cell size: {cell_w:.3f} x {cell_h:.3f}",
    ]
    draw.rectangle([(5, 5), (220, 5 + 16 * len(lines))], fill=(0, 0, 0, 160))
    for i, line in enumerate(lines):
        draw.text((10, 8 + 16 * i), line, fill=(255, 255, 255), font=font)
    return np.asarray(img)


def get_max_gap(env):
    scenario = env.base_env.scenario
    age = scenario.step_count[0] - scenario.last_seen[0]
    return float(age.max().item())


def get_fresh_ratio(env):
    scenario = env.base_env.scenario
    age = scenario.step_count[0] - scenario.last_seen[0]
    fresh = age <= float(scenario.revisit_limit)
    return float(fresh.float().mean().item())


def get_cell_size(env):
    scenario = env.base_env.scenario
    cell_w = scenario.width / scenario.grid_w
    cell_h = scenario.height / scenario.grid_h
    return float(cell_w), float(cell_h)


def main():
    parser = argparse.ArgumentParser(description="VMAS 覆盖任务可视化（阈值跳变版）")
    parser.add_argument("--model_path", default="models/coverage_vmas_policy_best.pth")
    parser.add_argument("--out_dir", default="videos")
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=200)
    # 与训练脚本保持一致的默认量纲：
    # 约 95km x 95km -> 仿真约 103 x 103（1.0仿真长度单位≈0.926km）
    parser.add_argument("--width", type=float, default=103.0)
    parser.add_argument("--height", type=float, default=103.0)
    parser.add_argument("--grid_w", type=int, default=10)
    parser.add_argument("--grid_h", type=int, default=10)
    parser.add_argument("--revisit_limit", type=int, default=10)
    parser.add_argument("--randomize_area", action="store_true")
    parser.add_argument("--width_min", type=float, default=95.0)
    parser.add_argument("--width_max", type=float, default=110.0)
    parser.add_argument("--height_min", type=float, default=95.0)
    parser.add_argument("--height_max", type=float, default=110.0)
    parser.add_argument("--speed_type_a_knots", type=float, default=30.0)
    parser.add_argument("--speed_type_b_knots", type=float, default=35.0)
    parser.add_argument("--sensor_range_type_a", type=float, default=27.0)
    parser.add_argument("--sensor_range_type_b", type=float, default=27.0)
    parser.add_argument("--coverage_margin", type=float, default=0.9)
    args = parser.parse_args()

    is_fork = multiprocessing.get_start_method() == "fork"
    device = torch.device(0) if torch.cuda.is_available() and not is_fork else torch.device("cpu")
    vmas_device = device

    speed_type_a = args.speed_type_a_knots / args.speed_type_a_knots
    speed_type_b = args.speed_type_b_knots / args.speed_type_a_knots
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
        speed_type_a=speed_type_a,
        speed_type_b=speed_type_b,
        sensor_range_type_a=args.sensor_range_type_a,
        sensor_range_type_b=args.sensor_range_type_b,
        render_style="binary",
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
            cell_w, cell_h = get_cell_size(env)
            frames.append(annotate_frame(frame, get_max_gap(env), get_fresh_ratio(env), cell_w, cell_h))

        with torch.no_grad():
            for _ in range(args.max_steps):
                td = policy(td)
                td = env.step(td)
                frame = render_frame(env)
                if frame is not None:
                    cell_w, cell_h = get_cell_size(env)
                    frames.append(annotate_frame(frame, get_max_gap(env), get_fresh_ratio(env), cell_w, cell_h))
                td_next = td.get("next")
                if td_next is not None and td_next.get("done").any().item():
                    break
                td = td_next

        out_path = os.path.join(args.out_dir, f"coverage_vmas_binary_episode_{ep + 1}.mp4")
        if frames:
            imageio.mimsave(out_path, frames, fps=args.fps)
            print(f"保存视频: {out_path} (frames={len(frames)})")

    env.close()


if __name__ == "__main__":
    main()
