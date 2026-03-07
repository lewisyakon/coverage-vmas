# VMAS 区域覆盖任务 MAPPO 算法详细分析报告

## 目录

1. [项目概述](#1-项目概述)
2. [场景设计详解](#2-场景设计详解)
3. [MAPPO模型架构详解](#3-mappo模型架构详解)
4. [训练流程详解](#4-训练流程详解)
5. [可视化渲染详解](#5-可视化渲染详解)
6. [关键算法分析](#6-关键算法分析)
7. [代码实现细节](#7-代码实现细节)
8. [参数配置汇总](#8-参数配置汇总)
9. [文件依赖与架构](#9-文件依赖与架构)

---

## 1. 项目概述

### 1.1 任务定义

本项目实现了一个**多智能体区域覆盖任务**（Multi-Agent Coverage Task），目标是在矩形区域内协调多个异构机器人，使整个区域的探测覆盖保持"新鲜"状态。

**核心挑战**：
- **协调问题**：多个Agent需要分工合作，避免重复覆盖同一区域
- **异构性**：Agent具有不同的速度-探测范围权衡
- **持续覆盖**：不仅要覆盖，还要定期重访已覆盖区域

### 1.2 技术栈

| 组件 | 技术选型 | 版本要求 |
|------|----------|----------|
| 仿真环境 | VMAS (Vectorized Multi-Agent Simulator) | - |
| 强化学习框架 | TorchRL | - |
| 深度学习框架 | PyTorch | - |
| 视频导出 | imageio | - |
| 数据处理 | NumPy, PIL | - |

### 1.3 算法选择：MAPPO

选择 **MAPPO (Multi-Agent Proximal Policy Optimization)** 的原因：

1. **多Agent适应性**：专为多智能体场景设计
2. **训练稳定性**：PPO的置信域优化保证稳定收敛
3. **集中式训练分散执行**：Critic可访问全局信息，Actor仅用局部观测
4. **异构Agent支持**：共享Actor可处理不同类型的Agent

---

## 2. 场景设计详解

### 2.1 文件结构

```python
scenario_coverage.py
│
├── AgentTypeSpec (数据类)          # Agent类型规格定义
├── Scenario (核心场景类)           # 继承BaseScenario
│   ├── __init__                   # 初始化参数
│   ├── make_world                 # 创建世界和Agent
│   ├── reset_world_at             # 重置环境
│   ├── _randomize_area            # 随机化区域大小
│   ├── _rebuild_grid              # 重建网格
│   ├── _update_coverage           # 更新覆盖状态
│   ├── post_step                  # 步后处理
│   ├── observation                # 观测函数
│   ├── reward                     # 奖励函数
│   └── extra_render               # 自定义渲染
```

### 2.2 Agent类型规格

```python
@dataclass
class AgentTypeSpec:
    name: str              # 类型名称
    max_speed: float       # 最大速度 (m/step)
    sensor_range: float    # 探测范围 (m)
    color: tuple           # 渲染颜色 (R, G, B)
```

**定义的两种Agent类型**：

```python
self.type_specs = [
    AgentTypeSpec(
        name="type_A",
        max_speed=1.0,      # 较慢
        sensor_range=0.18,  # 探测范围大
        color=Color.BLUE.value
    ),
    AgentTypeSpec(
        name="type_B",
        max_speed=1.5,      # 较快
        sensor_range=0.12,  # 探测范围小
        color=Color.GREEN.value
    )
]

self.agent_type_ids = [0, 0, 1, 1]  # 2个type_A + 2个type_B
```

### 2.3 物理世界配置

```python
world = World(
    batch_dim=60,           # 并行环境数
    device=device,          # 计算设备
    dt=1.0,                # 时间步长 (秒)
    drag=0.25,             # 空气阻力系数
    x_semidim=0.5,         # X方向半长 (区域宽=1.0)
    y_semidim=0.5,         # Y方向半长 (区域高=1.0)
    dim_c=0,               # 通信维度(无通信)
)
```

**物理参数说明**：
- `dt=1.0`：每步代表1秒模拟时间
- `drag=0.25`：速度衰减因子，模拟摩擦力
- `dim_c=0`：本场景不使用Agent间通信

### 2.4 Agent物理模型

```python
Agent(
    name=f"agent_{i}",
    collide=True,          # 启用碰撞检测
    mass=1.0,              # 质量 (kg)
    shape=Sphere(radius=0.02),  # 碰撞半径
    max_speed=spec.max_speed, # 最大速度限制
    u_range=1.0,           # 控制力范围
    color=spec.color       # 渲染颜色
)
```

### 2.5 网格覆盖系统

#### 2.5.1 网格划分

```python
# 10x10 网格划分
xs = (torch.arange(10) + 0.5) * (1.0 / 10) - 0.5
# 结果: [-0.45, -0.35, ..., 0.35, 0.45]
# 共10个网格中心点，范围 [-0.5, 0.5]

ys = (torch.arange(10) + 0.5) * (1.0 / 10) - 0.5

# 生成网格中心坐标
grid = torch.stack(torch.meshgrid(xs, ys, indexing="xy"), dim=-1)
# shape: (10, 10, 2)

self.grid_centers = grid.reshape(-1, 2)
# shape: (100, 2) - 100个网格中心
```

**网格属性计算**：
```python
cell_w = width / grid_w = 1.0 / 10 = 0.1  # 网格宽度
cell_h = height / grid_h = 1.0 / 10 = 0.1  # 网格高度
```

#### 2.5.2 覆盖状态追踪

```python
# last_seen: (batch_dim, 100) - 每个batch每个网格最后被探测的步数
self.last_seen = torch.full((batch_dim, 100), -1e9, device=device)

# step_count: (batch_dim,) - 当前步数
self.step_count = torch.zeros(batch_dim, device=device)
```

#### 2.5.3 覆盖更新算法

```python
def _update_coverage(self):
    # 1. 更新时间步
    self.step_count += 1.0

    # 2. 遍历所有Agent更新探测状态
    for agent in self.world.agents:
        pos = agent.state.pos  # (batch, 2)

        # 计算所有网格到Agent的距离
        diff = self.grid_centers.unsqueeze(0) - pos.unsqueeze(1)
        # diff shape: (batch, 100, 2)

        dist = torch.linalg.vector_norm(diff, dim=-1)
        # dist shape: (batch, 100)

        # 判断哪些网格在探测范围内
        covered = dist <= agent.sensor_range
        # covered shape: (batch, 100) - bool tensor

        # 更新last_seen
        step_t = self.step_count.unsqueeze(1).expand_as(self.last_seen)
        self.last_seen = torch.where(covered, step_t, self.last_seen)

    # 3. 计算年龄
    age = self.step_count.unsqueeze(1) - self.last_seen
    # age shape: (batch, 100) - 年龄越大表示越久未被探测
```

**覆盖判断图示**：

```
     Agent位置
         ●
         │
         │  sensor_range
         │  ◄───────
         │         │
         │   ┌─────┼─────┐
         │   │  ✓  │  ✓  │
         │   ├─────┼─────┤      网格状态:
         │   │  ✓  │  ●  │      ✓ = 已被探测
         │   └─────┼─────┘      ○ = 未被探测
         └─────────┘
```

### 2.6 奖励函数设计

#### 2.6.1 新鲜度计算

```python
# 归一化年龄
age_norm = age / revisit_limit  # 0=刚探测, 1=即将过期
age_norm = age_norm.clamp(0.0, 2.0)  # 限制范围

# 非线性曲线 (关键设计)
age_curve = torch.clamp(age_norm, 0.0, 1.0) ** age_curve_power
# age_curve_power = 3.0

# 新鲜度 = 1 - 曲线值
smooth_score = 1.0 - age_curve
```

**非线性曲线效果**：

| age_norm | age_curve (n=3) | fresh_score |
|----------|-----------------|-------------|
| 0.0      | 0.000           | 1.000       |
| 0.25     | 0.016           | 0.984       |
| 0.5      | 0.125           | 0.875       |
| 0.75     | 0.422           | 0.578       |
| 1.0      | 1.000           | 0.000       |

**设计意图**：
- 年龄在0~0.75区间时，惩罚较温和
- 年龄接近1.0时，惩罚急剧增加
- 激励Agent在网格"即将过期"前主动重访

#### 2.6.2 奖励组成

```python
# 1. 基础新鲜度 (平均新鲜度)
fresh_ratio = smooth_score.mean(dim=1)  # (batch,)

# 2. 改进奖励 (鼓励持续探索)
improve = fresh_ratio - prev_fresh_ratio

# 3. 最大年龄惩罚 (惩罚长期未覆盖区域)
max_age_norm = max_age / revisit_limit
max_age_norm = max_age_norm.clamp(0.0, 2.0)

# 4. 综合奖励
reward = fresh_ratio \
         + reward_improve_weight * improve \
         - max_age_penalty * max_age_norm
```

**奖励权重**：
```python
reward_improve_weight = 0.5   # 改进奖励权重
max_age_penalty = 0.2         # 最大年龄惩罚权重
```

### 2.7 观测空间设计

#### 2.7.1 观测向量组成

```python
# 观测维度分析
obs = torch.cat([
    pos_norm,      # 2维 - 归一化位置
    vel_norm,      # 2维 - 归一化速度
    type_one_hot,  # 2维 - Agent类型独热编码
    type_feats,    # 2维 - Agent能力特征
    global_feats,  # 3维 - 全局状态
    local_fresh,   # 1维 - 局部新鲜度
], dim=-1)
# 总计: 12维观测
```

#### 2.7.2 各分量详解

**位置归一化**：
```python
pos_norm_x = (pos_x + width/2) / width   # [0, 1]
pos_norm_y = (pos_y + height/2) / height  # [0, 1]
```

**速度归一化**：
```python
max_speed_global = max(t.max_speed for t in self.type_specs)  # 1.5
vel_norm = vel / max_speed_global  # 相对最大速度
```

**类型特征**：
```python
type_feats = [
    agent.max_speed / max_speed_global,    # 速度能力比例
    agent.sensor_range / max_range_global   # 探测范围比例
]
```

**全局特征**：
```python
global_feats = [
    fresh_ratio,                              # 0~1
    (max_age / revisit_limit).clamp(0, 2),   # 0~2
    step_count / max_steps                    # 0~1
]
```

**局部新鲜度**：
```python
# 计算Agent周围的3x3网格新鲜度均值
radius = max(cell_w, cell_h) * (local_radius_cells + 0.5)
local_mask = dist <= radius
local_fresh = masked_sum / masked_count
```

### 2.8 渲染系统

#### 2.8.1 Gradient模式 (渐变色)

```python
# 颜色插值公式
t = age_curve  # 0(最新) ~ 1(过期)

r = 0.2 + 0.6 * t  # 红色: 0.2 → 0.8
g = 0.8 - 0.6 * t  # 绿色: 0.8 → 0.2

# 透明度
alpha = 0.25
```

**颜色映射**：
| age_curve | RGB | 含义 |
|-----------|-----|------|
| 0.0 | (0.2, 0.8, 0.2) | 深绿 - 刚探测 |
| 0.5 | (0.5, 0.5, 0.2) | 黄色 - 中等 |
| 1.0 | (0.8, 0.2, 0.2) | 深红 - 即将过期 |

#### 2.8.2 Binary模式 (二值化)

```python
fresh = age <= revisit_limit  # 阈值判断

if fresh:
    box.set_color(0.2, 0.8, 0.2, 0.25)  # 绿色 - 有效
else:
    box.set_color(0.8, 0.2, 0.2, 0.10)  # 红色 - 过期
```

---

## 3. MAPPO模型架构详解

### 3.1 整体架构图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           MAPPO 架构                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                      Shared Actor (Policy)                       │    │
│  │  ┌───────────┐     ┌─────────────────┐     ┌───────────────┐   │    │
│  │  │ Observation│────▶│  MultiAgentMLP  │────▶│NormalParamExt │   │    │
│  │  │  (4, 12)  │     │  (256×2层)       │     │  (分离均值方差)  │   │    │
│  │  └───────────┘     └─────────────────┘     └───────────────┘   │    │
│  │                          │                                      │    │
│  │                          ▼                                      │    │
│  │                   ┌───────────┐                                  │    │
│  │                   │TanhNormal │  ⬅── 连续动作分布               │    │
│  │                   │  分布     │                                  │    │
│  │                   └─────┬─────┘                                  │    │
│  │                         │                                        │    │
│  │  ┌──────────────────────┼──────────────────────┐                │    │
│  │  │                      │                      │                │    │
│  │  ▼                      ▼                      ▼                │    │
│  │ action_loc          action_scale            action              │    │
│  │ (4, 2)               (4, 2)              连续控制力               │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                    ▲                                     │
│                                    │                                     │
│         ┌──────────────────────────┼──────────────────────────┐        │
│         │                          │                          │        │
│         ▼                          ▼                          ▼        │
│    ┌──────────┐              ┌──────────┐              ┌──────────┐    │
│    │ Agent 0  │              │ Agent 1  │              │ Agent 2  │    │
│    │  ● BLUE  │              │  ● BLUE  │              │ ● GREEN  │    │
│    │ type_A   │              │ type_A   │              │ type_B   │    │
│    │ v=1.0    │              │ v=1.0    │              │ v=1.5    │    │
│    │ r=0.18   │              │ r=0.18   │              │ r=0.12   │    │
│    └──────────┘              └──────────┘              └──────────┘    │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    Centralized Critic                            │    │
│  │  ┌───────────┐     ┌─────────────────┐     ┌───────────┐       │    │
│  │  │Concat Obs │────▶│  MultiAgentMLP  │────▶│   Value   │       │    │
│  │  │  (4, 12) │     │  (256×2层)       │     │   (4, 1)  │       │    │
│  │  └───────────┘     └─────────────────┘     └───────────┘       │    │
│  │                          ▲                                      │    │
│  │                          │                                      │    │
│  │         ┌───────────────┼───────────────┐                      │    │
│  │         │               │               │                      │    │
│  │         ▼               ▼               ▼                      │    │
│  │    ┌──────────┐    ┌──────────┐    ┌──────────┐                 │    │
│  │    │ V(s,0)   │    │ V(s,1)   │    │ V(s,2)   │                 │    │
│  │    │ 0.85     │    │ 0.72     │    │ 0.91     │                 │    │
│  │    └──────────┘    └──────────┘    └──────────┘                 │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Actor网络详解

#### 3.2.1 网络结构

```python
policy_net = torch.nn.Sequential(
    MultiAgentMLP(
        n_agent_inputs=12,           # 每个Agent的观测维度
        n_agent_outputs=4,           # 输出: 2×均值 + 2×标准差
        n_agents=4,                  # Agent数量
        centralised=False,           # 分散式决策
        share_params=True,           # 共享参数
        device=device,
        depth=2,                     # 2层隐藏层
        num_cells=256,              # 每层256神经元
        activation_class=torch.nn.Tanh,
    ),
    NormalParamExtractor(),         # 分离均值和标准差
)
```

#### 3.2.2 MultiAgentMLP内部结构

```
输入: (batch, agents, obs_dim) = (B, 4, 12)

Layer 1: Linear(12, 256) + Tanh
    ↓
Layer 2: Linear(256, 256) + Tanh
    ↓
输出: (B, 4, 4)  [2×mean + 2×std per agent]
```

#### 3.2.3 NormalParamExtractor

```python
# 输入: (B, 4, 4)  [loc×2, scale×2 per agent]
# 输出:
#   loc:     (B, 4, 2)  动作均值
#   scale:   (B, 4, 2)  动作标准差
```

#### 3.2.4 动作分布 (TanhNormal)

```python
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
```

**TanhNormal分布**：
- 先采样高斯分布 `N(μ, σ)`
- 再通过tanh函数映射到 [-1, 1] 区间
- 适合连续控制任务，避免大动作

### 3.3 Critic网络详解

```python
critic_net = MultiAgentMLP(
    n_agent_inputs=12,           # 每个Agent的观测维度
    n_agent_outputs=1,          # 单个Value输出
    n_agents=4,                  # Agent数量
    centralised=True,           # 集中式 Critic (能看到所有)
    share_params=True,           # 共享参数
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
```

**集中式Critic的优势**：
- Critic可以看到所有Agent的观测 `obs_all = (B, 4, 12)`
- 能够学习到Agent间的协作关系
- 提供更准确的价值估计

### 3.4 TensorDict数据结构

训练过程中使用TensorDict管理数据：

```python
# 轨迹数据形状
tensordict_data.shape  # (batch_dim, frames_per_batch, n_agents, ...)

# 关键字段
{
    ("agents", "observation"):    # (60, 200, 4, 12)  观测
    ("agents", "action"):          # (60, 200, 4, 2)   动作
    ("agents", "reward"):           # (60, 200, 4, 1)   奖励
    ("agents", "done"):            # (60, 200, 4, 1)   终止
    ("agents", "terminated"):       # (60, 200, 4, 1)   终止
    ("next", "agents", "done"):    # ...
    ("next", "agents", "terminated"):  # ...
}
```

---

## 4. 训练流程详解

### 4.1 训练流程图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           完整训练流程                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────┐                                                      │
│  │ 初始化环境    │                                                      │
│  │ build_env()   │                                                      │
│  └──────┬───────┘                                                      │
│         │                                                               │
│         ▼                                                               │
│  ┌──────────────┐                                                      │
│  │ 初始化网络    │                                                      │
│  │ build_policy()│                                                      │
│  │ build_critic()│                                                      │
│  └──────┬───────┘                                                      │
│         │                                                               │
│         ▼                                                               │
│  ┌──────────────┐                                                      │
│  │ 数据收集      │◀────────────────────────────────────────┐            │
│  │ SyncDataCollector                              │            │
│  │ 1. env.reset()                                   │            │
│  │ 2. policy(td) → action                           │            │
│  │ 3. env.step(td) → reward, done                   │            │
│  │ 4. 重复直到frames_per_batch                       │            │
│  └──────┬───────┘                                    │            │
│         │                                             │            │
│         ▼                                             │            │
│  ┌──────────────┐                                    │            │
│  │ 计算GAE      │                                    │            │
│  │ GAE.estimate()                                    │            │
│  │ δ = r + γV(s') - V(s)                             │            │
│  │ A = Σ(γλ)^t δ                                     │            │
│  └──────┬───────┘                                    │            │
│         │                                             │            │
│         ▼                                             │            │
│  ┌──────────────┐                                    │            │
│  │ 存入Buffer   │                                    │            │
│  │ ReplayBuffer│                                    │            │
│  └──────┬───────┘                                    │            │
│         │                                             │            │
│         ▼                                             │            │
│  ┌──────────────┐                                    │            │
│  │ PPO更新      │                                    │            │
│  │ for epoch:   │                                    │            │
│  │   for batch: │                                    │            │
│  │     loss.backward()                               │            │
│  │     optim.step()                                  │            │
│  └──────┬───────┘                                    │            │
│         │                                             │            │
│         ▼                                             │            │
│  ┌──────────────┐                                    │            │
│  │ 评估 & 保存   │                                    │            │
│  │ 计算episode_reward                                │            │
│  │ 保存best模型    ──────────────────────────────────┘            │
│  └──────────────┘                                                      │
│         │                                                               │
│         └─────────────────────────────────────────────────────────────▶│
│                  (循环 n_iters 次)                                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.2 数据收集器配置

```python
collector = SyncDataCollector(
    env,                              # VMAS环境
    policy,                           # 策略网络
    device=vmas_device,              # VMAS运行设备
    storing_device=device,            # 数据存储设备
    frames_per_batch=6000,           # 每批次收集6000帧
    total_frames=6000 * 1500,        # 总共900万帧
)
```

**收集逻辑**：
```python
for tensordict_data in collector:
    # tensordict_data 包含:
    # - observation: (60, 200, 4, 12)
    # - action: (60, 200, 4, 2)
    # - reward: (60, 200, 4, 1)
    # - done: (60, 200, 4, 1)
```

### 4.3 经验回放缓冲区

```python
replay_buffer = ReplayBuffer(
    storage=LazyTensorStorage(
        args.frames_per_batch,  # 6000帧
        device=device          # 存储设备
    ),
    sampler=SamplerWithoutReplacement(),  # 不重复采样
    batch_size=args.minibatch_size,       # 400
)
```

### 4.4 GAE优势估计

```python
# 配置损失模块
loss_module = ClipPPOLoss(
    actor_network=policy,
    critic_network=critic,
    clip_epsilon=0.2,
    entropy_coeff=1e-4,
    normalize_advantage=False,
)

# 设置GAE
loss_module.make_value_estimator(
    ValueEstimators.GAE,
    gamma=args.gamma,    # 0.99
    lmbda=args.lmbda,   # 0.9
)

# 计算GAE
with torch.no_grad():
    GAE(
        tensordict_data,
        params=loss_module.critic_network_params,
        target_params=loss_module.target_critic_network_params,
    )
```

**GAE公式**：
```
δ_t = r_t + γV(s_{t+1}) - V(s_t)

A_t = δ_t + (γλ)δ_{t+1} + (γλ)^2 δ_{t+2} + ... + (γλ)^{T-t} δ_T
    = Σ_{k=0}^{∞} (γλ)^k δ_{t+k}
```

### 4.5 PPO损失函数

```python
loss_vals = loss_module(subdata)
loss_value = (
    loss_vals["loss_objective"]     # 策略损失
    + loss_vals["loss_critic"]     # 价值损失
    + loss_vals["loss_entropy"]     # 熵正则
)
```

#### 4.5.1 策略损失 (Clipped Objective)

```python
# PPO Clip公式
ratio = π_θ(a|s) / π_θ_old(a|s)

loss_clip = -min(
    ratio * A,
    clip(ratio, 1-ε, 1+ε) * A
).mean()
```

**Clip机制图示**：
```
          │
     ratio│
          │    ┌─────────────────┐
          │    │                 │  A > 0: 鼓励有利动作
    1+ε ──┼────│ ╱╲               │  A < 0: 惩罚不利动作
          │    │╱  ╲              │  超出范围: 停止更新
    1-ε ──┼────│    ╲╱            │
          │    └─────────────────┘
          └───────────────────────────▶ A (优势)
                  0
```

#### 4.5.2 价值损失

```python
loss_critic = MSE(V(s), GAE_target)
```

#### 4.5.3 熵正则

```python
loss_entropy = -H(π)  # 鼓励探索
```

### 4.6 训练循环

```python
for tensordict_data in collector:  # 1500次迭代
    # 1. 填充done和terminated
    tensordict_data.set(("next", "agents", "done"), ...)
    tensordict_data.set(("next", "agents", "terminated"), ...)

    # 2. 计算GAE优势
    with torch.no_grad():
        GAE(tensordict_data, ...)

    # 3. 展平并存入Buffer
    data_view = tensordict_data.reshape(-1)
    replay_buffer.extend(data_view)

    # 4. PPO更新
    for _ in range(args.num_epochs):  # 30轮
        for _ in range(args.frames_per_batch // args.minibatch_size):
            subdata = replay_buffer.sample()  # 15个batch
            loss = loss_module(subdata)
            loss["loss"].backward()
            torch.nn.utils.clip_grad_norm_(loss_module.parameters(), 1.0)
            optim.step()
            optim.zero_grad()

    # 5. 更新策略权重
    collector.update_policy_weights_()

    # 6. 评估并保存最佳模型
    episode_reward_mean = calculate_reward(tensordict_data)
    if episode_reward_mean > best_score:
        save_best_model()
```

### 4.7 训练配置汇总

| 参数 | 值 | 说明 |
|------|-----|------|
| frames_per_batch | 6000 | 每批次收集的帧数 |
| n_iters | 1500 | 训练迭代次数 |
| num_epochs | 30 | PPO更新轮数 |
| minibatch_size | 400 | 每次更新的batch大小 |
| lr | 1e-4 | 学习率 |
| max_grad_norm | 1.0 | 梯度裁剪阈值 |
| clip_epsilon | 0.2 | PPO裁剪参数 |
| gamma | 0.99 | 折扣因子 |
| lmbda | 0.9 | GAE参数 |
| entropy_eps | 1e-4 | 熵系数 |
| num_envs | 60 | 并行环境数 |

---

## 5. 可视化渲染详解

### 5.1 渲染模式对比

#### 5.1.1 Gradient模式

**文件**: `visualize_coverage_vmas.py`

```python
scenario = Scenario(
    ...
    render_style="gradient",  # 默认渐变色
)
```

**视觉效果**：
- 网格颜色从 **绿色(新鲜)** → **红色(过期)** 渐变
- 颜色深浅表示覆盖的"新鲜度"
- 直观展示覆盖空洞和重访需求

#### 5.1.2 Binary模式

**文件**: `visualize_coverage_vmas_binary.py`

```python
scenario = Scenario(
    ...
    render_style="binary",  # 阈值跳变
)
```

**视觉效果**：
- **绿色**: age ≤ revisit_limit (10步内探测过)
- **红色**: age > revisit_limit (超过10步未探测)
- 更清晰的"有效/失效"边界

### 5.2 视频导出流程

```python
# 1. 初始化环境和策略
scenario = Scenario(...)
env = VmasEnv(scenario=scenario, ...)
policy = build_policy(env, device)
policy.load_state_dict(torch.load(model_path))

# 2. 逐episode录制
for ep in range(episodes):
    frames = []
    td = env.reset(seed=seed + ep)

    # 3. 渲染初始帧
    frame = env.render(mode="rgb_array")
    frames.append(annotate_frame(frame, ...))

    # 4. 交互并录制
    with torch.no_grad():
        for _ in range(max_steps):
            td = policy(td)           # 策略推理
            td = env.step(td)         # 环境步进
            frame = env.render()       # 获取帧
            frames.append(annotate_frame(...))

    # 5. 保存为MP4
    imageio.mimsave(out_path, frames, fps=30)
```

### 5.3 帧标注系统

```python
def annotate_frame(frame, max_gap, fresh_ratio, cell_w, cell_h):
    """在视频帧左上角添加信息标注"""

    lines = [
        f"Max gap: {max_gap:.1f} min",      # 最大未覆盖时间间隔
        f"Fresh ratio: {fresh_ratio:.1%}",  # 覆盖率
        f"Cell size: {cell_w:.3f} x {cell_h:.3f}",  # 网格大小
    ]

    # 绘制半透明背景
    draw.rectangle(
        [(5, 5), (220, 5 + 16 * len(lines))],
        fill=(0, 0, 0, 160)
    )

    # 绘制文字
    for i, line in enumerate(lines):
        draw.text((10, 8 + 16 * i), line, fill=(255, 255, 255))
```

### 5.4 关键指标获取

```python
def get_max_gap(env):
    """获取最大未覆盖时间间隔"""
    scenario = env.base_env.scenario
    age = scenario.step_count[0] - scenario.last_seen[0]
    return float(age.max().item())

def get_fresh_ratio(env):
    """获取覆盖率"""
    scenario = env.base_env.scenario
    age = scenario.step_count[0] - scenario.last_seen[0]
    fresh = age <= float(scenario.revisit_limit)
    return float(fresh.float().mean().item())
```

---

## 6. 关键算法分析

### 6.1 异构Agent处理策略

本项目通过**共享Actor + 类型特征**的方式处理异构Agent：

```python
# Actor网络对所有Agent共享参数
policy_net = MultiAgentMLP(
    ...
    share_params=True,  # 关键：所有Agent共享同一套参数
)

# Agent通过观测中的类型特征区分行为
type_one_hot = torch.zeros(2)
type_one_hot[type_id] = 1.0

type_feats = torch.tensor([
    agent.max_speed / max_speed_global,
    agent.sensor_range / max_range_global,
])
```

**优势**：
- 减少参数量
- 促进知识共享
- 允许Agent相互学习不同类型的策略

### 6.2 新鲜度曲线的设计动机

```python
age_curve_power = 3.0  # 三次方非线性

fresh = 1 - clamp(age_norm, 0, 1) ** 3
```

**设计原理**：

1. **线性问题**：
   - 线性新鲜度 `fresh = 1 - age_norm`
   - 激励均匀分布覆盖

2. **非线性优势**：
   - 在阈值附近惩罚急剧增加
   - 防止Agent忽视即将过期的区域
   - 更符合"定期巡逻"的实际需求

### 6.3 协调机制

**问题**：如何避免多个Agent重复覆盖同一区域？

**解决方案**：

1. **全局覆盖信息**：
   ```python
   global_feats = [
       fresh_ratio,      # 整体覆盖率
       max_age,          # 最久未覆盖区域
       step_count,       # 当前进度
   ]
   ```

2. **局部新鲜度提示**：
   ```python
   # Agent周围3x3网格的新鲜度
   local_fresh = masked_sum / masked_count
   ```

3. **奖励函数**：
   ```python
   # 改进奖励：鼓励持续提高覆盖率
   improve = fresh_ratio - prev_fresh_ratio
   ```

### 6.4 训练稳定性设计

| 设计 | 作用 |
|------|------|
| 梯度裁剪 | 防止梯度爆炸 |
| PPO Clip | 防止策略更新过大 |
| GAE | 稳定优势估计 |
| 熵正则 | 防止过早收敛 |
| 共享参数 | 增加训练数据量 |

---

## 7. 代码实现细节

### 7.1 环境构建函数

```python
def build_env(args, vmas_device):
    """构建VMAS环境"""

    # 1. 创建场景
    scenario = Scenario(
        width=args.width,           # 1.0
        height=args.height,         # 1.0
        grid_w=args.grid_w,         # 10
        grid_h=args.grid_h,         # 10
        revisit_limit=args.revisit_limit,  # 10
        max_steps=args.max_steps,   # 200
        randomize_area=args.randomize_area,  # False
        width_range=(args.width_min, args.width_max),
        height_range=(args.height_min, args.height_max),
        coverage_margin=args.coverage_margin,  # 0.9
    )

    # 2. 创建VMAS环境
    env = VmasEnv(
        scenario=scenario,
        num_envs=args.num_envs,      # 60
        continuous_actions=True,     # 连续动作
        max_steps=args.max_steps,    # 200
        device=vmas_device,
    )

    # 3. 添加奖励汇总变换
    env = TransformedEnv(
        env,
        RewardSum(
            in_keys=[env.reward_key],
            out_keys=[("agents", "episode_reward")]
        ),
    )

    return env
```

### 7.2 策略构建函数

```python
def build_policy(env, device):
    """构建策略网络"""

    # 1. 创建策略网络
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

    # 2. 包装为TensorDictModule
    policy_module = TensorDictModule(
        policy_net,
        in_keys=[("agents", "observation")],
        out_keys=[("agents", "loc"), ("agents", "scale")],
    )

    # 3. 创建概率Actor
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
```

### 7.3 Critic构建函数

```python
def build_critic(env, device):
    """构建Critic网络"""

    critic_net = MultiAgentMLP(
        n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1],
        n_agent_outputs=1,
        n_agents=env.n_agents,
        centralised=True,      # 集中式Critic
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
```

### 7.4 TensorDict操作

```python
# 设置done和terminated
tensordict_data.set(
    ("next", "agents", "done"),
    tensordict_data.get(("next", "done"))
    .unsqueeze(-1)
    .expand(tensordict_data.get_item_shape(("next", env.reward_key)))
)

# 展平数据用于ReplayBuffer
data_view = tensordict_data.reshape(-1)

# 获取数据
done_base = tensordict_data.get(("next", "done"))
ep_rew = tensordict_data.get(("next", "agents", "episode_reward"))
```

### 7.5 断点续训

```python
# 方式1：指定路径恢复
if args.resume_policy:
    policy.load_state_dict(torch.load(args.resume_policy))

# 方式2：恢复最佳模型
if args.resume_best:
    best_policy = os.path.join(args.save_dir, "coverage_vmas_policy_best.pth")
    best_critic = os.path.join(args.save_dir, "coverage_vmas_critic_best.pth")
    policy.load_state_dict(torch.load(best_policy))
    critic.load_state_dict(torch.load(best_critic))
```

---

## 8. 参数配置汇总

### 8.1 场景参数

| 参数 | 默认值 | 范围 | 说明 |
|------|--------|------|------|
| width | 1.0 | - | 区域宽度 |
| height | 1.0 | - | 区域高度 |
| grid_w | 10 | - | X方向网格数 |
| grid_h | 10 | - | Y方向网格数 |
| revisit_limit | 10 | - | 重访限制步数 |
| max_steps | 200 | - | 每回合最大步数 |
| randomize_area | False | - | 是否随机化区域大小 |
| width_range | (0.8, 1.2) | - | 宽度随机范围 |
| height_range | (0.8, 1.2) | - | 高度随机范围 |
| coverage_margin | 0.9 | - | 覆盖能力裕度 |
| reward_improve_weight | 0.5 | - | 改进奖励权重 |
| max_age_penalty | 0.2 | - | 最大年龄惩罚权重 |
| local_radius_cells | 1 | - | 局部半径(网格数) |
| age_curve_power | 3.0 | - | 年龄曲线幂次 |
| render_style | "gradient" | gradient/binary | 渲染风格 |

### 8.2 Agent类型参数

| 类型 | max_speed | sensor_range | color |
|------|------------|---------------|-------|
| type_A | 1.0 | 0.18 | BLUE |
| type_B | 1.5 | 0.12 | GREEN |

### 8.3 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| frames_per_batch | 6000 | 每批次帧数 |
| n_iters | 1500 | 迭代次数 |
| num_epochs | 30 | PPO更新轮数 |
| minibatch_size | 400 | Minibatch大小 |
| lr | 1e-4 | 学习率 |
| max_grad_norm | 1.0 | 梯度裁剪 |
| clip_epsilon | 0.2 | PPO裁剪 |
| gamma | 0.99 | 折扣因子 |
| lmbda | 0.9 | GAE参数 |
| entropy_eps | 1e-4 | 熵系数 |
| num_envs | 60 | 并行环境数 |
| seed | 0 | 随机种子 |

### 8.4 可视化参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| model_path | models/coverage_vmas_policy_best.pth | 模型路径 |
| out_dir | videos | 输出目录 |
| fps | 30 | 帧率 |
| episodes | 1 | 录制回合数 |

---

## 9. 文件依赖与架构

### 9.1 文件结构

```
coverage_mappo_vmas/
│
├── scenario_coverage.py          # 场景定义 (核心)
│   ├── AgentTypeSpec             # Agent类型数据类
│   └── Scenario                  # 场景主类
│
├── train_coverage_vmas_mappo.py  # 训练脚本 (核心)
│   ├── build_env()               # 构建环境
│   ├── build_policy()            # 构建策略
│   ├── build_critic()            # 构建Critic
│   └── train()                   # 训练循环
│
├── visualize_coverage_vmas.py    # 可视化(渐变色)
│   ├── annotate_frame()          # 帧标注
│   └── main()                    # 主函数
│
├── visualize_coverage_vmas_binary.py  # 可视化(二值化)
│   └── main()                        # 主函数
│
├── models/                       # 模型保存目录
│   ├── coverage_vmas_policy.pth
│   ├── coverage_vmas_policy_best.pth
│   ├── coverage_vmas_critic.pth
│   └── coverage_vmas_critic_best.pth
│
└── videos/                      # 视频保存目录
    ├── coverage_vmas_episode_*.mp4
    └── coverage_vmas_binary_episode_*.mp4
```

### 9.2 依赖关系图

```
                    ┌─────────────────────┐
                    │  VMAS 仿真环境       │
                    │ (torchrl.envs.libs) │
                    └──────────┬──────────┘
                               │
                               ▼
┌───────────────────────────────────────────────────────────────┐
│                      依赖层次                                  │
├───────────────────────────────────────────────────────────────┤
│                                                              │
│   visualize_coverage_vmas.py                                  │
│         │                                                    │
│         │ imports                                            │
│         ▼                                                    │
│   ┌─────────────────────────────────────────────────────┐   │
│   │ train_coverage_vmas_mappo.py                         │   │
│   │         │                                           │   │
│   │         │ imports                                   │   │
│   │         ▼                                           │   │
│   │   ┌─────────────────────────────────────────────┐   │   │
│   │   │   scenario_coverage.py                      │   │   │
│   │   │   - Scenario (场景类)                       │   │   │
│   │   │   - AgentTypeSpec (Agent类型)               │   │   │
│   │   └─────────────────────────────────────────────┘   │   │
│   │             ▲                       │              │   │
│   │             │                       │              │   │
│   │    torchrl  │              torchrl  │              │   │
│   │   ─────────┼────────────────────┼─────             │   │
│   │             │                    │                │   │
│   │             ▼                    ▼                │   │
│   │   ┌──────────────┐      ┌────────────────┐           │   │
│   │   │ VmasEnv      │      │ Probabilistic  │           │   │
│   │   │ TransformedEnv│      │ Actor/Critic   │           │   │
│   │   └──────────────┘      └────────────────┘           │   │
│   │                                                    │   │
│   └────────────────────────────────────────────────────┘   │
│                                                              │
└───────────────────────────────────────────────────────────────┘
```

### 9.3 核心类继承关系

```
BaseScenario (VMAS)
    │
    └── Scenario
        │
        ├── make_world()
        ├── reset_world_at()
        ├── _randomize_area()
        ├── _rebuild_grid()
        ├── _update_coverage()
        ├── post_step()
        ├── observation()
        ├── reward()
        └── extra_render()
```

### 9.4 启动方式

**训练**：
```bash
python train_coverage_vmas_mappo.py --n_iters 1500 --num_envs 60
```

**可视化 (渐变色)**：
```bash
python visualize_coverage_vmas.py --model_path models/coverage_vmas_policy_best.pth
```

**可视化 (二值化)**：
```bash
python visualize_coverage_vmas_binary.py --model_path models/coverage_vmas_policy_best.pth
```

---

## 附录：关键数学公式

### A. 新鲜度计算

```
age_norm = clamp(age / revisit_limit, 0, 2)

age_curve = clamp(age_norm, 0, 1) ^ age_curve_power

fresh_score = 1 - age_curve
```

### B. GAE优势估计

```
δ_t = r_t + γV(s_{t+1}) - V(s_t)

A_t = Σ_{k=0}^{∞} (γλ)^k δ_{t+k}
```

### C. PPO Clip损失

```
ratio = π_θ(a|s) / π_θ_old(a|s)

L_clip = E[min(
    ratio * A_t,
    clamp(ratio, 1-ε, 1+ε) * A_t
)]
```

### D. 总损失

```
L = L_clip + c_1 * L_VF + c_2 * S[π_θ]

其中：
- L_VF = MSE(V_θ, V_target)  # 价值损失
- S = -Σ π(s) log π(s)       # 熵
- c_1 = 1                    # 价值损失权重
- c_2 = entropy_eps          # 熵权重
```

---

**报告生成时间**: 2026-02-09

**分析文件**:
- `scenario_coverage.py` - 场景定义 (323行)
- `train_coverage_vmas_mappo.py` - 训练脚本 (283行)
- `visualize_coverage_vmas.py` - 可视化 (157行)
- `visualize_coverage_vmas_binary.py` - 二值化可视化 (159行)
