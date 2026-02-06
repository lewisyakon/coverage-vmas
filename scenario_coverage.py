#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VMAS 自定义场景：矩形区域覆盖 + 重访约束
两种 agent 类型，各 2 个，差异仅速度与探测范围
"""

from dataclasses import dataclass
import torch
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
        local_radius_cells=1,
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
        self.local_radius_cells = int(local_radius_cells)

        self.type_specs = [
            AgentTypeSpec("type_A", max_speed=1.0, sensor_range=0.18, color=Color.BLUE.value),
            AgentTypeSpec("type_B", max_speed=1.5, sensor_range=0.12, color=Color.GREEN.value),
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
        fresh = age <= float(self.revisit_limit)
        self._cached_fresh_ratio = fresh.float().mean(dim=1)
        self._cached_max_age = age.max(dim=1).values
        improve = self._cached_fresh_ratio - prev_fresh
        max_age_norm = self._cached_max_age / max(1.0, float(self.revisit_limit))
        max_age_norm = max_age_norm.clamp(0.0, 2.0)
        self._cached_reward = (
            self._cached_fresh_ratio
            + self.reward_improve_weight * improve
            - self.max_age_penalty * max_age_norm
        )
        self._prev_fresh_ratio = prev_fresh

    def post_step(self):
        self._update_coverage()

    def observation(self, agent: Agent):
        # 观测：位置、速度、类型onehot、类型能力、全局覆盖信息
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

        # 局部覆盖提示：以自身为中心的 3x3 近邻覆盖新鲜度均值
        cell_w = self.width / self.grid_w
        cell_h = self.height / self.grid_h
        radius = max(cell_w, cell_h) * (self.local_radius_cells + 0.5)
        diff = self.grid_centers.unsqueeze(0) - pos.unsqueeze(1)
        dist = torch.linalg.vector_norm(diff, dim=-1)
        local_mask = dist <= radius
        age = self.step_count.unsqueeze(1) - self.last_seen
        fresh = (age <= float(self.revisit_limit)).float()
        masked = fresh * local_mask.float()
        local_sum = masked.sum(dim=1)
        local_cnt = local_mask.sum(dim=1).clamp_min(1.0)
        local_fresh = (local_sum / local_cnt).unsqueeze(-1)

        obs = torch.cat(
            [pos_norm, vel_norm, type_one_hot, type_feats, global_feats, local_fresh],
            dim=-1,
        )
        return obs

    def reward(self, agent: Agent):
        return self._cached_reward

    def extra_render(self, env_index: int = 0):
        # 在渲染中标出“10分钟内覆盖过”的网格
        from vmas.simulator import rendering

        geoms = []
        cell_w = self.width / self.grid_w
        cell_h = self.height / self.grid_h

        age = self.step_count[env_index] - self.last_seen[env_index]
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
        return geoms
