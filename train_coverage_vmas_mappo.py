#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VMAS 覆盖任务 MAPPO（方案A：共享Actor + 集中式Critic）
"""

import argparse
import logging
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
    # 探测范围映射说明：
    # - 文档口径：A/B 两型无人艇对海探测范围均为 [2,25] km
    # - 因当前 demo 为归一化场景，先用同一仿真探测半径表示“能力一致”
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
        reward_improve_weight=args.reward_improve_weight,
        max_age_penalty=args.max_age_penalty,
        oldest_repair_weight=args.oldest_repair_weight,
        oldest_k_ratio=args.oldest_k_ratio,
        overlap_penalty_weight=args.overlap_penalty_weight,
        same_type_separation_weight=args.same_type_separation_weight,
        same_type_min_dist_ratio=args.same_type_min_dist_ratio,
        speed_type_a=speed_type_a,
        speed_type_b=speed_type_b,
        sensor_range_type_a=args.sensor_range_type_a,
        sensor_range_type_b=args.sensor_range_type_b,
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
        # 提高数值稳定性：限制最小方差，避免分布尺度过小/异常导致采样不稳定
        NormalParamExtractor(scale_lb=1e-3),
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
    # 避免 torchrl 的 INFO 日志打断 tqdm 单行进度条显示
    logging.getLogger("torchrl").setLevel(logging.WARNING)

    env = build_env(args, vmas_device)
    policy = build_policy(env, device)
    critic = build_critic(env, device)
    steps_per_batch = args.frames_per_batch // args.num_envs if args.num_envs > 0 else args.frames_per_batch
    if steps_per_batch < args.max_steps:
        print(
            "[提示] 当前每批步数小于单回合长度："
            f"steps_per_batch={steps_per_batch}, max_steps={args.max_steps}。"
            "若直接记录批末episode_reward会出现固定交替。"
        )
        print(
            "[建议] 可将 --frames_per_batch 设为 num_envs*max_steps 的倍数。"
            f"例如当前可用: {args.num_envs * args.max_steps}。"
        )

    def try_load_state_dict(module, ckpt_path, module_name):
        if not ckpt_path or not os.path.exists(ckpt_path):
            return False
        try:
            module.load_state_dict(torch.load(ckpt_path, map_location=device))
            print(f"[恢复] 已加载{module_name}: {ckpt_path}")
            return True
        except RuntimeError as e:
            print(f"[恢复] 跳过{module_name}加载（结构不兼容，通常是观测维度变更）: {ckpt_path}")
            print(f"[恢复] 详细原因: {e}")
            return False

    if args.resume_policy or args.resume_critic:
        if not try_load_state_dict(policy, args.resume_policy, "策略模型"):
            print("[恢复] 未找到策略模型，改为从头训练")
        if not try_load_state_dict(critic, args.resume_critic, "价值模型"):
            print("[恢复] 未找到价值模型，改为从头训练")
    elif args.resume_best:
        best_policy = os.path.join(args.save_dir, "coverage_vmas_policy_best.pth")
        best_critic = os.path.join(args.save_dir, "coverage_vmas_critic_best.pth")
        ok_p = try_load_state_dict(policy, best_policy, "最佳策略模型")
        ok_c = try_load_state_dict(critic, best_critic, "最佳价值模型")
        if not (ok_p and ok_c):
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
        normalize_advantage=True,
        # 多智能体下避免跨 agent 维做归一化统计，减少训练抖动
        normalize_advantage_exclude_dims=(-2, -1),
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

    pbar = tqdm(total=args.n_iters, desc="episode_reward_mean = 0", dynamic_ncols=True, mininterval=0.5)
    best_score = -float("inf")
    last_complete_episode_mean = None
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
                # 防炸保护：若 loss 非有限值，跳过本次更新
                if not torch.isfinite(loss_value):
                    optim.zero_grad(set_to_none=True)
                    continue
                loss_value.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    loss_module.parameters(), args.max_grad_norm, error_if_nonfinite=False
                )
                if isinstance(grad_norm, torch.Tensor) and not torch.isfinite(grad_norm):
                    optim.zero_grad(set_to_none=True)
                    continue
                has_bad_grad = False
                for p in loss_module.parameters():
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        has_bad_grad = True
                        break
                if has_bad_grad:
                    optim.zero_grad(set_to_none=True)
                    continue
                optim.step()
                optim.zero_grad(set_to_none=True)

        collector.update_policy_weights_()

        done_base = tensordict_data.get(("next", "done"))
        ep_rew = tensordict_data.get(("next", "agents", "episode_reward"))
        if done_base.ndim == 3:
            done_base = done_base.squeeze(-1)
        done_any = bool(done_base.any().item())
        if done_any:
            steps = done_base.shape[1]
            t_idx = torch.arange(steps, device=done_base.device).unsqueeze(0)
            last_done_idx = (done_base.float() * t_idx).max(dim=1).values.long()
            env_idx = torch.arange(done_base.shape[0], device=done_base.device)
            end_rew = ep_rew[env_idx, last_done_idx]
            no_done = ~done_base.any(dim=1)
            if no_done.any():
                end_rew[no_done] = ep_rew[no_done, -1]
            last_complete_episode_mean = end_rew.mean().item()
            episode_reward_mean = last_complete_episode_mean
        else:
            # 无完整回合结束时，沿用最近一次“完整回合”的均值，避免半回合值造成固定高低交替
            if last_complete_episode_mean is None:
                episode_reward_mean = ep_rew[:, -1].mean().item()
            else:
                episode_reward_mean = last_complete_episode_mean

        if done_any and episode_reward_mean > best_score:
            best_score = episode_reward_mean
            os.makedirs(args.save_dir, exist_ok=True)
            best_policy_path = os.path.join(args.save_dir, "coverage_vmas_policy_best.pth")
            best_critic_path = os.path.join(args.save_dir, "coverage_vmas_critic_best.pth")
            torch.save(policy.state_dict(), best_policy_path)
            torch.save(critic.state_dict(), best_critic_path)
            tqdm.write(
                f"[NEW BEST] episode_reward_mean={best_score:.3f} | "
                f"saved: {best_policy_path}, {best_critic_path}"
            )
        pbar.set_description(f"episode_reward_mean = {episode_reward_mean:.3f}", refresh=True)
        pbar.set_postfix(best=f"{best_score:.3f}", refresh=False)
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
    # 二值奖励+大场景下梯度波动更大，默认学习率下调提升稳定性
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--clip_epsilon", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lmbda", type=float, default=0.9)
    parser.add_argument("--entropy_eps", type=float, default=1e-4)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--num_envs", type=int, default=60)
    # 区域尺寸按“探测上限 25km”重标定（1 step≈1min, A艇30节=>0.926km/min 对应 speed=1.0）：
    # - 1.0 仿真长度单位 ≈ 0.926 km
    # - 无人艇最大探测半径 25km => 25 / 0.926 ≈ 27.0 仿真单位
    # - 4艘艇瞬时理论覆盖上限约 4 * pi * 25^2 = 7852 km^2（不计重叠）
    # - 10分钟动态扫掠上限（直线近似）：
    #   单艇 A_10min ≈ pi*r^2 + 2*r*L, 其中 L≈(0.926~1.08)*10 ≈ 9.3~10.8 km
    #   取 L≈10km 时单艇约 2463 km^2，4艇总计约 9850 km^2（不计重叠）
    # 因此默认海域设置为约 95km x 95km（约 8570 km^2）：
    # - 明显大于瞬时上限 7852 km^2（避免过易）
    # - 低于动态上限 9850 km^2（在协同良好时可行）
    parser.add_argument("--width", type=float, default=103.0)
    parser.add_argument("--height", type=float, default=103.0)
    parser.add_argument("--randomize_area", action="store_true")
    parser.add_argument("--width_min", type=float, default=95.0)
    parser.add_argument("--width_max", type=float, default=110.0)
    parser.add_argument("--height_min", type=float, default=95.0)
    parser.add_argument("--height_max", type=float, default=110.0)
    parser.add_argument("--speed_type_a_knots", type=float, default=30.0)
    parser.add_argument("--speed_type_b_knots", type=float, default=35.0)
    # A/B 两型无人艇文档中探测范围同为 [2,25]km。
    # 这里按“上限 25km”配置探测半径：25 / 0.926 ≈ 27.0（仿真单位）
    parser.add_argument("--sensor_range_type_a", type=float, default=27.0)
    parser.add_argument("--sensor_range_type_b", type=float, default=27.0)
    # 奖励项权重：强化“修复最老区域”，抑制固定区域摆动
    parser.add_argument("--reward_improve_weight", type=float, default=0.5)
    parser.add_argument("--max_age_penalty", type=float, default=0.5)
    parser.add_argument("--oldest_repair_weight", type=float, default=0.35)
    parser.add_argument("--oldest_k_ratio", type=float, default=0.12)
    # 防止同类抱团/探测圈重叠过大
    parser.add_argument("--overlap_penalty_weight", type=float, default=0.10)
    parser.add_argument("--same_type_separation_weight", type=float, default=0.12)
    parser.add_argument("--same_type_min_dist_ratio", type=float, default=0.40)
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
