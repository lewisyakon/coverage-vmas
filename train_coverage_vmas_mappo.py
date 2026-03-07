#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VMAS 覆盖任务 MAPPO（方案A：共享Actor + 集中式Critic）
"""

import argparse
import os

import torch
from tensordict.nn import set_composite_lp_aggregate, TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import multiprocessing
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import RewardSum, TransformedEnv
from torchrl.envs.libs.vmas import VmasEnv
from torchrl.modules import MultiAgentMLP, ProbabilisticActor, TanhNormal
from torchrl.objectives import ClipPPOLoss, ValueEstimators
from tqdm import tqdm

from scenario_coverage import Scenario


def build_env(args, vmas_device):
    # 无人艇速度按文档比例映射：
    # A:[0,30]节, B:[0,35]节 -> 仿真 max_speed 比例 1.0 : (35/30)=1.1666667
    #
    # 实际尺寸换算约定（用于注释说明）：
    # - 任务层时间映射：1 step ≈ 1 minute
    # - A艇 30节 = 55.56 km/h = 0.926 km/min，对应仿真 speed=1.0
    # - 因此 1.0 仿真长度单位 ≈ 0.926 km（近似）
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
    )
    env = VmasEnv(
        scenario=scenario,
        num_envs=args.num_envs,
        continuous_actions=True,
        max_steps=args.max_steps,
        device=vmas_device,
    )
    env = TransformedEnv(
        env,
        RewardSum(in_keys=[env.reward_key], out_keys=[("agents", "episode_reward")]),
    )
    return env


def build_policy(env, device):
    policy_net = torch.nn.Sequential(
        MultiAgentMLP(
            n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1],
            n_agent_outputs=2 * env.full_action_spec[env.action_key].shape[-1],
            n_agents=env.n_agents,
            centralised=False,
            share_params=True,
            device=device,
            depth=2,
            num_cells=256,
            activation_class=torch.nn.Tanh,
        ),
        NormalParamExtractor(),
    )
    policy_module = TensorDictModule(
        policy_net,
        in_keys=[("agents", "observation")],
        out_keys=[("agents", "loc"), ("agents", "scale")],
    )
    policy = ProbabilisticActor(
        module=policy_module,
        spec=env.action_spec_unbatched,
        in_keys=[("agents", "loc"), ("agents", "scale")],
        out_keys=[env.action_key],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "low": env.full_action_spec_unbatched[env.action_key].space.low,
            "high": env.full_action_spec_unbatched[env.action_key].space.high,
        },
        return_log_prob=True,
    )
    return policy


def build_critic(env, device):
    critic_net = MultiAgentMLP(
        n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1],
        n_agent_outputs=1,
        n_agents=env.n_agents,
        centralised=True,
        share_params=True,
        device=device,
        depth=2,
        num_cells=256,
        activation_class=torch.nn.Tanh,
    )
    critic = TensorDictModule(
        module=critic_net,
        in_keys=[("agents", "observation")],
        out_keys=[("agents", "state_value")],
    )
    return critic


def train(args):
    is_fork = multiprocessing.get_start_method() == "fork"
    device = torch.device(0) if torch.cuda.is_available() and not is_fork else torch.device("cpu")
    vmas_device = device

    torch.manual_seed(args.seed)
    set_composite_lp_aggregate(False).set()

    env = build_env(args, vmas_device)
    policy = build_policy(env, device)
    critic = build_critic(env, device)
    if args.resume_policy or args.resume_critic:
        if args.resume_policy and os.path.exists(args.resume_policy):
            policy.load_state_dict(torch.load(args.resume_policy, map_location=device))
            print(f"[恢复] 已加载策略模型: {args.resume_policy}")
        else:
            print("[恢复] 未找到策略模型，改为从头训练")
        if args.resume_critic and os.path.exists(args.resume_critic):
            critic.load_state_dict(torch.load(args.resume_critic, map_location=device))
            print(f"[恢复] 已加载价值模型: {args.resume_critic}")
        else:
            print("[恢复] 未找到价值模型，改为从头训练")
    elif args.resume_best:
        best_policy = os.path.join(args.save_dir, "coverage_vmas_policy_best.pth")
        best_critic = os.path.join(args.save_dir, "coverage_vmas_critic_best.pth")
        if os.path.exists(best_policy) and os.path.exists(best_critic):
            policy.load_state_dict(torch.load(best_policy, map_location=device))
            critic.load_state_dict(torch.load(best_critic, map_location=device))
            print(f"[恢复] 已加载最佳模型: {best_policy}, {best_critic}")
        else:
            print("[恢复] 未找到最佳模型，改为从头训练")

    collector = SyncDataCollector(
        env,
        policy,
        device=vmas_device,
        storing_device=device,
        frames_per_batch=args.frames_per_batch,
        total_frames=args.frames_per_batch * args.n_iters,
    )

    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(args.frames_per_batch, device=device),
        sampler=SamplerWithoutReplacement(),
        batch_size=args.minibatch_size,
    )

    loss_module = ClipPPOLoss(
        actor_network=policy,
        critic_network=critic,
        clip_epsilon=args.clip_epsilon,
        entropy_coeff=args.entropy_eps,
        normalize_advantage=False,
    )
    loss_module.set_keys(
        reward=env.reward_key,
        action=env.action_key,
        value=("agents", "state_value"),
        done=("agents", "done"),
        terminated=("agents", "terminated"),
    )
    loss_module.make_value_estimator(ValueEstimators.GAE, gamma=args.gamma, lmbda=args.lmbda)
    GAE = loss_module.value_estimator

    optim = torch.optim.Adam(loss_module.parameters(), args.lr)

    pbar = tqdm(total=args.n_iters, desc="episode_reward_mean = 0")
    best_score = -float("inf")
    for tensordict_data in collector:
        tensordict_data.set(
            ("next", "agents", "done"),
            tensordict_data.get(("next", "done"))
            .unsqueeze(-1)
            .expand(tensordict_data.get_item_shape(("next", env.reward_key))),
        )
        tensordict_data.set(
            ("next", "agents", "terminated"),
            tensordict_data.get(("next", "terminated"))
            .unsqueeze(-1)
            .expand(tensordict_data.get_item_shape(("next", env.reward_key))),
        )

        with torch.no_grad():
            GAE(
                tensordict_data,
                params=loss_module.critic_network_params,
                target_params=loss_module.target_critic_network_params,
            )

        data_view = tensordict_data.reshape(-1)
        replay_buffer.extend(data_view)

        for _ in range(args.num_epochs):
            for _ in range(args.frames_per_batch // args.minibatch_size):
                subdata = replay_buffer.sample()
                loss_vals = loss_module(subdata)
                loss_value = (
                    loss_vals["loss_objective"]
                    + loss_vals["loss_critic"]
                    + loss_vals["loss_entropy"]
                )
                loss_value.backward()
                torch.nn.utils.clip_grad_norm_(loss_module.parameters(), args.max_grad_norm)
                optim.step()
                optim.zero_grad()

        collector.update_policy_weights_()

        done_base = tensordict_data.get(("next", "done"))
        ep_rew = tensordict_data.get(("next", "agents", "episode_reward"))
        if done_base.ndim == 3:
            done_base = done_base.squeeze(-1)
        if done_base.any():
            steps = done_base.shape[1]
            t_idx = torch.arange(steps, device=done_base.device).unsqueeze(0)
            last_done_idx = (done_base.float() * t_idx).max(dim=1).values.long()
            env_idx = torch.arange(done_base.shape[0], device=done_base.device)
            end_rew = ep_rew[env_idx, last_done_idx]
            no_done = ~done_base.any(dim=1)
            if no_done.any():
                end_rew[no_done] = ep_rew[no_done, -1]
            episode_reward_mean = end_rew.mean().item()
        else:
            episode_reward_mean = ep_rew[:, -1].mean().item()

        if episode_reward_mean > best_score:
            best_score = episode_reward_mean
            os.makedirs(args.save_dir, exist_ok=True)
            torch.save(policy.state_dict(), os.path.join(args.save_dir, "coverage_vmas_policy_best.pth"))
            torch.save(critic.state_dict(), os.path.join(args.save_dir, "coverage_vmas_critic_best.pth"))
        pbar.set_description(f"episode_reward_mean = {episode_reward_mean:.3f}", refresh=False)
        pbar.update()

    os.makedirs(args.save_dir, exist_ok=True)
    torch.save(policy.state_dict(), os.path.join(args.save_dir, "coverage_vmas_policy.pth"))
    torch.save(critic.state_dict(), os.path.join(args.save_dir, "coverage_vmas_critic.pth"))


def main():
    parser = argparse.ArgumentParser(description="VMAS 覆盖任务 MAPPO")
    parser.add_argument("--frames_per_batch", type=int, default=6000)
    parser.add_argument("--n_iters", type=int, default=1500)
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--minibatch_size", type=int, default=400)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--clip_epsilon", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lmbda", type=float, default=0.9)
    parser.add_argument("--entropy_eps", type=float, default=1e-4)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--num_envs", type=int, default=60)
    # 区域尺寸联动调整说明：
    # 旧速度配置平均值: (1.0 + 1.5)/2 = 1.25
    # 新速度配置平均值: (1.0 + 1.1666667)/2 = 1.08333335
    # 为保持接近的任务难度，将默认区域按 1.0833/1.25 ≈ 0.8667 缩放。
    #
    # 对应实际海域大小（按上面的近似换算）：
    # - 默认 width=height=0.87 -> 边长约 0.87 * 0.926 = 0.806 km
    #   即约 0.806 km × 0.806 km，面积约 0.65 km^2
    # - 随机范围 [0.70, 1.04] -> 边长约 [0.648, 0.963] km
    #   即面积约 [0.42, 0.93] km^2
    parser.add_argument("--width", type=float, default=0.87)
    parser.add_argument("--height", type=float, default=0.87)
    parser.add_argument("--randomize_area", action="store_true")
    parser.add_argument("--width_min", type=float, default=0.70)
    parser.add_argument("--width_max", type=float, default=1.04)
    parser.add_argument("--height_min", type=float, default=0.70)
    parser.add_argument("--height_max", type=float, default=1.04)
    parser.add_argument("--speed_type_a_knots", type=float, default=30.0)
    parser.add_argument("--speed_type_b_knots", type=float, default=35.0)
    parser.add_argument("--coverage_margin", type=float, default=0.9)
    parser.add_argument("--grid_w", type=int, default=10)
    parser.add_argument("--grid_h", type=int, default=10)
    parser.add_argument("--revisit_limit", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_dir", type=str, default="models")
    parser.add_argument("--resume_best", action="store_true")
    parser.add_argument("--resume_policy", type=str, default="")
    parser.add_argument("--resume_critic", type=str, default="")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
