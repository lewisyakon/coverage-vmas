#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VMAS 覆盖任务可视化（连续覆盖口径）：
- 不使用“单网格是否被覆盖”的离散判据
- 使用高分辨率连续采样点评估覆盖新鲜度
"""

import argparse
import os

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch import multiprocessing
from torchrl.envs import RewardSum, TransformedEnv
from torchrl.envs.libs.vmas import VmasEnv

from scenario_coverage import Scenario
from train_coverage_vmas_mappo import build_policy


class ContinuousCoverageTracker:
    def __init__(self, width, height, sample_w, sample_h, revisit_limit):
        self.width = float(width)
        self.height = float(height)
        self.sample_w = int(sample_w)
        self.sample_h = int(sample_h)
        self.revisit_limit = float(revisit_limit)
        xs = (np.arange(self.sample_w, dtype=np.float32) + 0.5) * (self.width / self.sample_w) - self.width / 2.0
        ys = (np.arange(self.sample_h, dtype=np.float32) + 0.5) * (self.height / self.sample_h) - self.height / 2.0
        gx, gy = np.meshgrid(xs, ys, indexing="xy")
        self.points = np.stack([gx, gy], axis=-1).reshape(-1, 2)  # (N,2)
        self.last_seen = np.full((self.points.shape[0],), -self.revisit_limit, dtype=np.float32)

    def update(self, agent_positions, agent_sensor_ranges, step_t):
        for i in range(agent_positions.shape[0]):
            px, py = float(agent_positions[i, 0]), float(agent_positions[i, 1])
            r = float(agent_sensor_ranges[i])
            dx = self.points[:, 0] - px
            dy = self.points[:, 1] - py
            covered = (dx * dx + dy * dy) <= (r * r)
            self.last_seen[covered] = float(step_t)

    def get_age(self, step_t):
        return np.maximum(0.0, float(step_t) - self.last_seen)

    def get_fresh_ratio(self, step_t):
        age = self.get_age(step_t)
        return float((age <= self.revisit_limit).mean())

    def get_max_gap(self, step_t):
        age = self.get_age(step_t)
        return float(age.max())

    def get_smooth_map(self, step_t, age_curve_power):
        age = self.get_age(step_t).reshape(self.sample_h, self.sample_w)
        age_norm = np.clip(age / max(1.0, self.revisit_limit), 0.0, 2.0)
        age_curve = np.clip(age_norm, 0.0, 1.0) ** float(age_curve_power)
        smooth = 1.0 - age_curve
        return np.clip(smooth, 0.0, 1.0)

    def get_binary_map(self, step_t):
        age = self.get_age(step_t).reshape(self.sample_h, self.sample_w)
        return (age <= self.revisit_limit).astype(np.float32)


def set_fixed_camera(env, padding_ratio=0.02):
    base_env = env.base_env
    scenario = base_env.scenario
    device = base_env.device
    half_w = scenario.width * 0.5 * (1.0 + padding_ratio)
    half_h = scenario.height * 0.5 * (1.0 + padding_ratio)
    target_half = max(float(half_w), float(half_h))
    scenario.viewer_zoom = max(1e-6, target_half ** 0.5)
    scenario.render_origin = torch.zeros(2, device=device)


def render_frame(env):
    set_fixed_camera(env)
    try:
        frame = env.render(mode="rgb_array")
    except TypeError:
        frame = env.render()
    return frame


def overlay_binary_map(frame, fresh_binary_map, alpha=0.22):
    # fresh_binary_map: 1=新鲜(绿), 0=过期(红)
    # 采样网格 y 轴向上为正，而图像像素 y 轴向下为正，需纵向翻转对齐渲染坐标
    fresh_binary_map = np.flipud(fresh_binary_map)
    r = (0.2 + 0.6 * (1.0 - fresh_binary_map)) * 255.0
    g = (0.8 - 0.6 * (1.0 - fresh_binary_map)) * 255.0
    b = np.full_like(r, 0.2 * 255.0)
    a = np.full_like(r, float(alpha) * 255.0)
    rgba = np.stack([r, g, b, a], axis=-1).astype(np.uint8)

    base = Image.fromarray(frame).convert("RGBA")
    heat = Image.fromarray(rgba, mode="RGBA").resize((base.width, base.height), Image.Resampling.BILINEAR)
    comp = Image.alpha_composite(base, heat)
    return np.asarray(comp.convert("RGB"))


def annotate_frame(frame, max_gap, fresh_ratio, sample_w, sample_h):
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    lines = [
        f"Max gap: {max_gap:.1f} min",
        f"Fresh ratio: {fresh_ratio:.1%}",
        f"Continuous samples: {sample_w} x {sample_h}",
    ]
    draw.rectangle([(5, 5), (260, 5 + 16 * len(lines))], fill=(0, 0, 0, 160))
    for i, line in enumerate(lines):
        draw.text((10, 8 + 16 * i), line, fill=(255, 255, 255), font=font)
    return np.asarray(img)


def get_agent_states(env):
    scenario = env.base_env.scenario
    positions = []
    ranges = []
    for agent in scenario.world.agents:
        p = agent.state.pos[0]
        positions.append([float(p[0].item()), float(p[1].item())])
        ranges.append(float(getattr(agent, "sensor_range", 0.0)))
    return np.asarray(positions, dtype=np.float32), np.asarray(ranges, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="VMAS 连续覆盖可视化 (二值叠层版)")
    parser.add_argument("--model_path", default="models/coverage_vmas_policy_best.pth")
    parser.add_argument("--out_dir", default="videos")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=200)
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
    parser.add_argument("--sample_w", type=int, default=120)
    parser.add_argument("--sample_h", type=int, default=120)
    parser.add_argument("--overlay_alpha", type=float, default=0.22)
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
        render_style="none",
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
        cur_scenario = env.base_env.scenario
        tracker = ContinuousCoverageTracker(
            width=cur_scenario.width,
            height=cur_scenario.height,
            sample_w=args.sample_w,
            sample_h=args.sample_h,
            revisit_limit=cur_scenario.revisit_limit,
        )

        positions, ranges = get_agent_states(env)
        tracker.update(positions, ranges, step_t=0.0)
        frame = render_frame(env)
        if frame is not None:
            fresh_map = tracker.get_binary_map(step_t=0.0)
            frame = overlay_binary_map(frame, fresh_map, alpha=args.overlay_alpha)
            frame = annotate_frame(
                frame,
                max_gap=tracker.get_max_gap(0.0),
                fresh_ratio=tracker.get_fresh_ratio(0.0),
                sample_w=args.sample_w,
                sample_h=args.sample_h,
            )
            frames.append(frame)

        with torch.no_grad():
            for _ in range(args.max_steps):
                td = policy(td)
                td = env.step(td)
                td_next = td.get("next")
                step_t = float(env.base_env.scenario.step_count[0].item())
                positions, ranges = get_agent_states(env)
                tracker.update(positions, ranges, step_t=step_t)

                frame = render_frame(env)
                if frame is not None:
                    fresh_map = tracker.get_binary_map(step_t=step_t)
                    frame = overlay_binary_map(frame, fresh_map, alpha=args.overlay_alpha)
                    frame = annotate_frame(
                        frame,
                        max_gap=tracker.get_max_gap(step_t),
                        fresh_ratio=tracker.get_fresh_ratio(step_t),
                        sample_w=args.sample_w,
                        sample_h=args.sample_h,
                    )
                    frames.append(frame)
                if td_next is not None and td_next.get("done").any().item():
                    break
                td = td_next

        out_path = os.path.join(args.out_dir, f"coverage_vmas_continuous_episode_{ep + 1}.mp4")
        if frames:
            imageio.mimsave(out_path, frames, fps=args.fps)
            print(f"保存视频: {out_path} (frames={len(frames)})")

    env.close()


if __name__ == "__main__":
    main()

