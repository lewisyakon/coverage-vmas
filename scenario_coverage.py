#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VMAS 自定义场景：矩形区域覆盖 + 重访约束
两种 agent 类型，各 2 个，差异仅速度与探测范围
"""

from dataclasses import dataclass
import torch
import torch.nn.functional as F
from vmas.simulator.core import Agent, Sphere, World
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color, ScenarioUtils


@dataclass
class AgentTypeSpec:
    name: str
    max_speed: float
    sensor_range: float
    color: tuple


class Scenario(BaseScenario):
    def __init__(
        self,
        width=1.0,
        height=1.0,
        grid_w=10,
        grid_h=10,
        revisit_limit=10,
        max_steps=200,
        randomize_area=False,
        width_range=(0.8, 1.2),
        height_range=(0.8, 1.2),
        coverage_margin=0.9,
        reward_improve_weight=0.5,
        max_age_penalty=0.2,
        overlap_penalty_weight=0.10,
        same_type_separation_weight=0.12,
        same_type_min_dist_ratio=0.40,
        local_radius_cells=1,
        age_curve_power=3.0,
        render_style="gradient",  # "gradient" | "binary" | "none"
        speed_type_a=1.0,
        speed_type_b=1.1666667,
        sensor_range_type_a=0.18,
        sensor_range_type_b=0.18,
    ):
        super().__init__()
        self.width = float(width)
        self.height = float(height)
        self.grid_w = int(grid_w)
        self.grid_h = int(grid_h)
        self.revisit_limit = int(revisit_limit)
        self.max_steps = int(max_steps)
        self.randomize_area = bool(randomize_area)
        self.width_range = (float(width_range[0]), float(width_range[1]))
        self.height_range = (float(height_range[0]), float(height_range[1]))
        self.coverage_margin = float(coverage_margin)
        self.reward_improve_weight = float(reward_improve_weight)
        self.max_age_penalty = float(max_age_penalty)
        self.overlap_penalty_weight = float(overlap_penalty_weight)
        self.same_type_separation_weight = float(same_type_separation_weight)
        self.same_type_min_dist_ratio = float(same_type_min_dist_ratio)
        self.local_radius_cells = int(local_radius_cells)
        self.age_curve_power = float(age_curve_power)
        self.render_style = str(render_style)
        self.speed_type_a = float(speed_type_a)
        self.speed_type_b = float(speed_type_b)
        self.sensor_range_type_a = float(sensor_range_type_a)
        self.sensor_range_type_b = float(sensor_range_type_b)

        self.type_specs = [
            # 速度映射说明（按比例）：
            # 无人艇A:[0,30]节, 无人艇B:[0,35]节
            # 仿真中采用 A:B = 30:35 = 1.0:1.1666667
            # 探测范围映射说明：
            # 文档口径下 A/B 两型无人艇均为 [2,25]km，demo 中默认设为相同探测半径
            AgentTypeSpec("type_A", max_speed=self.speed_type_a, sensor_range=self.sensor_range_type_a, color=Color.BLUE.value),
            AgentTypeSpec("type_B", max_speed=self.speed_type_b, sensor_range=self.sensor_range_type_b, color=Color.GREEN.value),
        ]
        self.agent_type_ids = [0, 0, 1, 1]

        self.grid_centers = None
        self.last_seen = None
        self.step_count = None
        self._cached_reward = None
        self._cached_fresh_ratio = None
        self._cached_max_age = None
        self._prev_fresh_ratio = None
        self._device = None

    def make_world(self, batch_dim: int, device: torch.device, **kwargs) -> World:
        self._device = device
        world = World(
            batch_dim,
            device,
            dt=1.0,
            drag=0.25,
            x_semidim=self.width / 2.0,
            y_semidim=self.height / 2.0,
            dim_c=0,
        )

        # agents
        for i, type_id in enumerate(self.agent_type_ids):
            spec = self.type_specs[type_id]
            agent = Agent(
                name=f"agent_{i}",
                collide=True,
                mass=1.0,
                shape=Sphere(radius=0.02),
                max_speed=spec.max_speed,
                u_range=1.0,
                color=spec.color,
            )
            agent.agent_index = i
            agent.type_id = type_id
            agent.sensor_range = spec.sensor_range
            world.add_agent(agent)

        self._rebuild_grid()

        self.last_seen = torch.full(
            (batch_dim, self.grid_centers.shape[0]),
            -1e9,
            device=device,
            dtype=torch.float32,
        )
        self.step_count = torch.zeros(batch_dim, device=device, dtype=torch.float32)
        self._cached_reward = torch.zeros(batch_dim, device=device, dtype=torch.float32)
        self._cached_fresh_ratio = torch.zeros(batch_dim, device=device, dtype=torch.float32)
        self._cached_max_age = torch.zeros(batch_dim, device=device, dtype=torch.float32)
        self._prev_fresh_ratio = torch.zeros(batch_dim, device=device, dtype=torch.float32)
        return world

    def reset_world_at(self, env_index: int | None = None):
        if env_index is None and self.randomize_area:
            self._randomize_area()
            self._rebuild_grid()
        ScenarioUtils.spawn_entities_randomly(
            self.world.agents,
            self.world,
            env_index,
            min_dist_between_entities=0.05,
            x_bounds=(-self.width / 2.0, self.width / 2.0),
            y_bounds=(-self.height / 2.0, self.height / 2.0),
        )
        if env_index is None:
            self.last_seen.fill_(-1e9)
            self.step_count.zero_()
            self._prev_fresh_ratio.zero_()
        else:
            self.last_seen[env_index].fill_(-1e9)
            self.step_count[env_index] = 0.0
            self._prev_fresh_ratio[env_index] = 0.0
        # 避免初始 age 过大导致数值不稳定
        if env_index is None:
            self.last_seen.fill_(-float(self.revisit_limit))
        else:
            self.last_seen[env_index].fill_(-float(self.revisit_limit))

    def _randomize_area(self):
        # 简单可行性判定：覆盖能力 >= 需求面积 * margin
        max_attempts = 50
        for _ in range(max_attempts):
            w = torch.empty(1).uniform_(self.width_range[0], self.width_range[1]).item()
            h = torch.empty(1).uniform_(self.height_range[0], self.height_range[1]).item()
            total_cover = 0.0
            for spec in self.type_specs:
                total_cover += 3.14159 * (spec.sensor_range ** 2)
            total_cover *= self.revisit_limit
            if total_cover >= (w * h * self.coverage_margin):
                self.width = w
                self.height = h
                # World semidims 是只读属性，需写入内部字段
                self.world._x_semidim = w / 2.0
                self.world._y_semidim = h / 2.0
                return
        # 退化：使用当前宽高
        return

    def _rebuild_grid(self):
        device = self._device or self.world.device
        # world 坐标以 (0,0) 为中心，网格中心也需居中
        xs = (torch.arange(self.grid_w, device=device) + 0.5) * (self.width / self.grid_w) - self.width / 2.0
        ys = (torch.arange(self.grid_h, device=device) + 0.5) * (self.height / self.grid_h) - self.height / 2.0
        grid = torch.stack(torch.meshgrid(xs, ys, indexing="xy"), dim=-1)
        self.grid_centers = grid.reshape(-1, 2)
        if self.last_seen is not None:
            self.last_seen = torch.full(
                (self.world.batch_dim, self.grid_centers.shape[0]),
                -1e9,
                device=device,
                dtype=torch.float32,
            )

    def _update_coverage(self):
        # 更新时间步
        self.step_count += 1.0

        prev_fresh = self._cached_fresh_ratio.clone()
        for agent in self.world.agents:
            pos = agent.state.pos  # (batch, 2)
            diff = self.grid_centers.unsqueeze(0) - pos.unsqueeze(1)
            dist = torch.linalg.vector_norm(diff, dim=-1)
            covered = dist <= agent.sensor_range
            step_t = self.step_count.unsqueeze(1).expand_as(self.last_seen)
            self.last_seen = torch.where(covered, step_t, self.last_seen)

        age = self.step_count.unsqueeze(1) - self.last_seen
        # 渐进式新鲜度惩罚：接近阈值时惩罚更大（而非二值跳变）
        age_norm = (age / max(1.0, float(self.revisit_limit))).clamp(0.0, 2.0)
        age_curve = torch.clamp(age_norm, 0.0, 1.0) ** self.age_curve_power
        smooth_score = 1.0 - age_curve
        self._cached_fresh_ratio = smooth_score.mean(dim=1)
        self._cached_max_age = age.max(dim=1).values
        improve = self._cached_fresh_ratio - prev_fresh
        max_age_norm = self._cached_max_age / max(1.0, float(self.revisit_limit))
        max_age_norm = max_age_norm.clamp(0.0, 2.0)
        # 编队去重约束：
        # 1) overlap_penalty: 任意两机探测圈过度重叠时惩罚
        # 2) same_type_penalty: 同类型平台距离过近时惩罚
        overlap_penalty = torch.zeros(self.world.batch_dim, device=self.world.device, dtype=torch.float32)
        same_type_penalty = torch.zeros(self.world.batch_dim, device=self.world.device, dtype=torch.float32)
        overlap_pairs = 0
        same_type_pairs = 0
        n_agents = len(self.world.agents)
        for i in range(n_agents):
            ai = self.world.agents[i]
            pi = ai.state.pos
            ri = float(getattr(ai, "sensor_range", 0.0))
            for j in range(i + 1, n_agents):
                aj = self.world.agents[j]
                pj = aj.state.pos
                rj = float(getattr(aj, "sensor_range", 0.0))
                sum_r = max(1e-6, ri + rj)
                dij = torch.linalg.vector_norm(pi - pj, dim=-1)
                overlap_penalty = overlap_penalty + ((sum_r - dij) / sum_r).clamp(0.0, 1.0)
                overlap_pairs += 1
                if getattr(ai, "type_id", -1) == getattr(aj, "type_id", -2):
                    min_dist = max(1e-6, self.same_type_min_dist_ratio * sum_r)
                    same_type_penalty = same_type_penalty + ((min_dist - dij) / min_dist).clamp(0.0, 1.0)
                    same_type_pairs += 1

        if overlap_pairs > 0:
            overlap_penalty = overlap_penalty / float(overlap_pairs)
        if same_type_pairs > 0:
            same_type_penalty = same_type_penalty / float(same_type_pairs)

        self._cached_reward = (
            self._cached_fresh_ratio
            + self.reward_improve_weight * improve
            - self.max_age_penalty * max_age_norm
            - self.overlap_penalty_weight * overlap_penalty
            - self.same_type_separation_weight * same_type_penalty
        )
        self._prev_fresh_ratio = prev_fresh

    def post_step(self):
        self._update_coverage()

    def observation(self, agent: Agent):
        # 观测：自身状态 + 队友相对状态 + 全局覆盖信息（含粗粒度热图）
        pos = agent.state.pos
        vel = agent.state.vel

        pos_norm = torch.stack(
            [
                (pos[:, 0] + self.width / 2.0) / self.width,
                (pos[:, 1] + self.height / 2.0) / self.height,
            ],
            dim=-1,
        )

        max_speed_global = max(t.max_speed for t in self.type_specs)
        max_range_global = max(t.sensor_range for t in self.type_specs)
        vel_norm = vel / max_speed_global

        type_one_hot = torch.zeros(
            (self.world.batch_dim, len(self.type_specs)),
            device=self.world.device,
        )
        type_one_hot[:, agent.type_id] = 1.0

        type_feats = torch.tensor(
            [
                agent.max_speed / max_speed_global,
                agent.sensor_range / max_range_global,
            ],
            device=self.world.device,
            dtype=torch.float32,
        ).unsqueeze(0).expand(self.world.batch_dim, -1)

        global_feats = torch.stack(
            [
                self._cached_fresh_ratio,
                (self._cached_max_age / max(1.0, float(self.revisit_limit))).clamp(0.0, 2.0),
                self.step_count / max(1.0, float(self.max_steps)),
            ],
            dim=-1,
        )

        # 局部覆盖提示：以自身为中心的 3x3 近邻平滑新鲜度均值
        cell_w = self.width / self.grid_w
        cell_h = self.height / self.grid_h
        radius = max(cell_w, cell_h) * (self.local_radius_cells + 0.5)
        diff = self.grid_centers.unsqueeze(0) - pos.unsqueeze(1)
        dist = torch.linalg.vector_norm(diff, dim=-1)
        local_mask = dist <= radius
        age = self.step_count.unsqueeze(1) - self.last_seen
        age_norm = (age / max(1.0, float(self.revisit_limit))).clamp(0.0, 2.0)
        smooth = 1.0 - (torch.clamp(age_norm, 0.0, 1.0) ** self.age_curve_power)
        masked = smooth * local_mask.float()
        local_sum = masked.sum(dim=1)
        local_cnt = local_mask.sum(dim=1).clamp_min(1.0)
        local_fresh = (local_sum / local_cnt).unsqueeze(-1)

        # 额外全局信息1：队友相对位置/速度（帮助分工，避免同类抱团）
        self_idx = int(getattr(agent, "agent_index", int(agent.name.split("_")[-1])))
        rel_feats = []
        for other in self.world.agents:
            other_idx = int(getattr(other, "agent_index", int(other.name.split("_")[-1])))
            if other_idx == self_idx:
                continue
            rel_pos = other.state.pos - pos
            rel_pos_norm = torch.stack(
                [rel_pos[:, 0] / max(1e-6, self.width), rel_pos[:, 1] / max(1e-6, self.height)],
                dim=-1,
            )
            rel_vel = (other.state.vel - vel) / max(1e-6, max_speed_global)
            rel_feats.append(torch.cat([rel_pos_norm, rel_vel], dim=-1))
        if rel_feats:
            teammate_rel = torch.cat(rel_feats, dim=-1)
        else:
            teammate_rel = torch.zeros((self.world.batch_dim, 0), device=self.world.device, dtype=torch.float32)

        # 额外全局信息2：全局覆盖热图（10x10 -> 5x5 平均池化，25维）
        fresh_map = smooth.reshape(self.world.batch_dim, self.grid_h, self.grid_w).unsqueeze(1)
        pooled = F.adaptive_avg_pool2d(fresh_map, (5, 5)).squeeze(1)
        global_heatmap = pooled.reshape(self.world.batch_dim, -1)

        obs = torch.cat(
            [
                pos_norm,
                vel_norm,
                type_one_hot,
                type_feats,
                global_feats,
                local_fresh,
                teammate_rel,
                global_heatmap,
            ],
            dim=-1,
        )
        return obs

    def reward(self, agent: Agent):
        return self._cached_reward

    def extra_render(self, env_index: int = 0):
        # 在渲染中标出“10分钟内覆盖过”的网格
        from vmas.simulator import rendering
        import math

        geoms = []
        cell_w = self.width / self.grid_w
        cell_h = self.height / self.grid_h

        age = self.step_count[env_index] - self.last_seen[env_index]
        if self.render_style == "none":
            # 不渲染网格方块，仅渲染各 agent 探测范围圈
            num_pts = 40
            for agent in self.world.agents:
                pos = agent.state.pos[env_index]
                cx, cy = float(pos[0].item()), float(pos[1].item())
                r = float(getattr(agent, "sensor_range", 0.0))
                if r <= 0:
                    continue
                pts = [
                    (
                        cx + r * math.cos(2.0 * math.pi * k / num_pts),
                        cy + r * math.sin(2.0 * math.pi * k / num_pts),
                    )
                    for k in range(num_pts)
                ]
                circ = rendering.make_polygon(pts)
                col = getattr(agent, "color", (0.9, 0.9, 0.1))
                circ.set_color(float(col[0]), float(col[1]), float(col[2]), 0.08)
                geoms.append(circ)
            return geoms
        if self.render_style == "binary":
            # 阈值跳变：10分钟内绿色，否则红色（仅影响渲染，不影响训练/观测）
            fresh = age <= float(self.revisit_limit)
            fresh = fresh.reshape(self.grid_h, self.grid_w)
            for gy in range(self.grid_h):
                for gx in range(self.grid_w):
                    x0 = -self.width / 2.0 + gx * cell_w
                    y0 = -self.height / 2.0 + gy * cell_h
                    x1 = x0 + cell_w
                    y1 = y0 + cell_h
                    box = rendering.make_polygon(
                        [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
                    )
                    if bool(fresh[gy, gx].item()):
                        box.set_color(0.2, 0.8, 0.2, 0.25)
                    else:
                        box.set_color(0.8, 0.2, 0.2, 0.10)
                    geoms.append(box)

            # 标注各 agent 的探测范围（浅色半透明圆）
            num_pts = 40
            for agent in self.world.agents:
                pos = agent.state.pos[env_index]
                cx, cy = float(pos[0].item()), float(pos[1].item())
                r = float(getattr(agent, "sensor_range", 0.0))
                if r <= 0:
                    continue
                pts = [
                    (
                        cx + r * math.cos(2.0 * math.pi * k / num_pts),
                        cy + r * math.sin(2.0 * math.pi * k / num_pts),
                    )
                    for k in range(num_pts)
                ]
                circ = rendering.make_polygon(pts)
                # 使用 agent 本身颜色，但更浅、更透明
                col = getattr(agent, "color", (0.9, 0.9, 0.1))
                circ.set_color(float(col[0]), float(col[1]), float(col[2]), 0.08)
                geoms.append(circ)
            return geoms
        age_norm = (age / max(1.0, float(self.revisit_limit))).clamp(0.0, 2.0)
        age_curve = torch.clamp(age_norm, 0.0, 1.0) ** self.age_curve_power
        age_curve = age_curve.reshape(self.grid_h, self.grid_w)

        for gy in range(self.grid_h):
            for gx in range(self.grid_w):
                x0 = -self.width / 2.0 + gx * cell_w
                y0 = -self.height / 2.0 + gy * cell_h
                x1 = x0 + cell_w
                y1 = y0 + cell_h
                box = rendering.make_polygon(
                    [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
                )
                t = float(age_curve[gy, gx].item())
                r = 0.2 + 0.6 * t
                g = 0.8 - 0.6 * t
                box.set_color(r, g, 0.2, 0.25)
                geoms.append(box)

        # 标注各 agent 的探测范围（浅色半透明圆）
        num_pts = 40
        for agent in self.world.agents:
            pos = agent.state.pos[env_index]
            cx, cy = float(pos[0].item()), float(pos[1].item())
            r = float(getattr(agent, "sensor_range", 0.0))
            if r <= 0:
                continue
            pts = [
                (
                    cx + r * math.cos(2.0 * math.pi * k / num_pts),
                    cy + r * math.sin(2.0 * math.pi * k / num_pts),
                )
                for k in range(num_pts)
            ]
            circ = rendering.make_polygon(pts)
            col = getattr(agent, "color", (0.9, 0.9, 0.1))
            circ.set_color(float(col[0]), float(col[1]), float(col[2]), 0.08)
            geoms.append(circ)
        return geoms
