# VMAS 区域覆盖任务 - MAPPO 实现

使用 VMAS 自定义场景实现多智能体区域覆盖巡逻任务，协调异构机器人使整个区域保持"新鲜"状态。

## 任务概述

### 目标
在矩形区域内协调多个异构机器人，使整个区域的探测覆盖保持"新鲜"状态。

### 核心挑战
- **异构性**：Agent 具有不同的速度-探测范围权衡
- **持续覆盖**：不仅要覆盖，还要定期重访已覆盖区域
- **协调分工**：避免多个 Agent 重复覆盖同一区域

### Agent 配置

| 数量 | 类型 | 最大速度 | 探测范围 | 颜色 |
|:----:|:----:|:--------:|:--------:|:----:|
| 2 | type_A | 慢(1.0) | 大(0.18) | 蓝色 |
| 2 | type_B | 快(1.5) | 小(0.12) | 绿色 |

### 环境参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 区域大小 | 1.0×1.0 | 归一化矩形覆盖区域 |
| 网格划分 | 10×10 | 100个探测单元 |
| 重访限制 | 10 | 网格"过期"阈值(步数) |
| 最大步数 | 200 | 每回合上限 |

## 运行

### 训练

```bash
python train_coverage_vmas_mappo.py
```

如果希望每次训练随机矩形大小，并保证"可扫完"的可行性：

```bash
python train_coverage_vmas_mappo.py --randomize_area --width_min 0.8 --width_max 1.2 --height_min 0.8 --height_max 1.2 --coverage_margin 0.9
```

### 可视化（MP4）

```bash
python visualize_coverage_vmas.py --model_path models/coverage_vmas_policy_best.pth --out_dir videos --episodes 1
```

说明：可视化会用绿色半透明标出"已覆盖"的网格区域。

二值化渲染模式：
```bash
python visualize_coverage_vmas_binary.py --model_path models/coverage_vmas_policy_best.pth --out_dir videos --episodes 1
```

## 参数说明

```bash
python train_coverage_vmas_mappo.py \
  --width 1.0 --height 1.0 \
  --grid_w 10 --grid_h 10 \
  --revisit_limit 10 \
  --max_steps 200 \
  --num_envs 60
```

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| width | 1.0 | 区域宽度 |
| height | 1.0 | 区域高度 |
| grid_w | 10 | X方向网格数 |
| grid_h | 10 | Y方向网格数 |
| revisit_limit | 10 | 重访限制步数 |
| max_steps | 200 | 每回合最大步数 |
| num_envs | 60 | 并行环境数 |
| n_iters | 1500 | 训练迭代次数 |
| lr | 1e-4 | 学习率 |

### 随机区域增强（可选）

```bash
python train_coverage_vmas_mappo.py --randomize_area --width_min 0.8 --width_max 1.2 --height_min 0.8 --height_max 1.2 --coverage_margin 0.9
```

增加环境多样性，提高模型泛化能力。

## 核心算法设计

### MAPPO 架构

| 组件 | 配置 |
|------|------|
| 算法 | Multi-Agent PPO (CTDE) |
| Actor | 共享策略，参数共享 |
| Critic | 集中式 Critic |
| 动作空间 | 连续动作 (TanhNormal) |

### 奖励函数

```python
reward = fresh_ratio
         + 0.5 * improve       # 改进奖励
         - 0.2 * max_age_norm  # 过期惩罚
```

### 非线性新鲜度惩罚

```python
age_norm = age / revisit_limit
age_curve = clamp(age_norm, 0, 1) ** 3.0  # 三次方
fresh_score = 1 - age_curve
```

**效果**：接近阈值时惩罚急剧增加，强制 Agent 定期巡逻。

## 文件结构

```
coverage_vmas/
├── scenario_coverage.py              # 场景定义
├── train_coverage_vmas_mappo.py      # 训练脚本
├── visualize_coverage_vmas.py       # 可视化(渐变色)
├── visualize_coverage_vmas_binary.py # 可视化(二值化)
├── README.md                         # 本文档
├── demo_presentation_script.md      # Demo 讲稿
└── hierarchical_central_actor_design.md  # 分层设计文档(未来扩展)
```

## 模型输出

- `models/coverage_vmas_policy.pth` / `models/coverage_vmas_critic.pth`（最终）
- `models/coverage_vmas_policy_best.pth` / `models/coverage_vmas_critic_best.pth`（最佳）
