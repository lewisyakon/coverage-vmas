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
| 2 | type_A (慢速型) | 30节 (1.0) | 25km (0.18) | 蓝色 |
| 2 | type_B (快速型) | 35节 (1.17) | 25km (0.18) | 绿色 |

> **速度映射说明**：无人艇A航速30节，无人艇B航速35节，仿真中采用 A:B = 1.0:1.1666667 的比例关系。

> **探测范围映射**：根据文档，A/B两型无人艇对海探测范围均为 [2,25] km，仿真中默认设为相同探测半径。

### 环境参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 区域大小 | 103×103 | 归一化矩形覆盖区域（约等于 95km×95km 真实海域） |
| 网格划分 | 10×10 | 100个探测单元 |
| 重访限制 | 10 | 网格"过期"阈值(步数，约10分钟) |
| 最大步数 | 200 | 每回合上限（约200分钟） |

> **时间映射**：1 step ≈ 1 minute，1.0 仿真长度单位 ≈ 0.926 km

## 运行

### 训练

```bash
python train_coverage_vmas_mappo.py
```

如果希望每次训练随机矩形大小，并保证"可扫完"的可行性：

```bash
python train_coverage_vmas_mappo.py --randomize_area --width_min 95 --width_max 110 --height_min 95 --height_max 110 --coverage_margin 0.9
```

### 推荐训练配置

以下是经过调优的推荐训练命令，适用于大多数场景：

```bash
python train_coverage_vmas_mappo.py \
  --frames_per_batch 12000 \
  --randomize_area \
  --width_min 95 --width_max 110 \
  --height_min 95 --height_max 110 \
  --speed_type_a_knots 30 --speed_type_b_knots 35 \
  --sensor_range_type_a 27 --sensor_range_type_b 27 \
  --revisit_limit 10 \
  --reward_improve_weight 0.5 \
  --max_age_penalty 0.5 \
  --oldest_repair_weight 0.35 \
  --oldest_k_ratio 0.12 \
  --overlap_penalty_weight 0.0 \
  --same_type_separation_weight 0.0 \
  --same_type_min_dist_ratio 0.40 \
  --save_dir /home/lmx/code/RL_Benchmark/coverage_vmas/models
```

#### 推荐配置参数详解

| 参数 | 值 | 说明 |
|------|-----|------|
| frames_per_batch | 12000 | 每批次收集的帧数。增大至 12000（默认6000），可减少策略更新频率，提高训练稳定性 |
| randomize_area | True | 开启区域大小随机化，增强模型泛化能力 |
| width_min/width_max | 95~110 | 区域宽度在 95~110 之间随机（约 88~102 km） |
| height_min/height_max | 95~110 | 区域高度在 95~110 之间随机 |
| speed_type_a_knots | 30 | A型艇速度 30节 |
| speed_type_b_knots | 35 | B型艇速度 35节 |
| sensor_range_type_a | 27 | A型艇探测半径 27（对应约 25km） |
| sensor_range_type_b | 27 | B型艇探测半径 27（对应约 25km） |
| revisit_limit | 10 | 网格过期阈值为 10 步（约 10 分钟） |
| reward_improve_weight | 0.5 | 改进奖励权重，鼓励持续改善覆盖 |
| max_age_penalty | 0.5 | 最久未覆盖惩罚权重（相比早期版本 0.2 提升至 0.5，强化定期巡逻） |
| oldest_repair_weight | 0.35 | 最老区域修复奖励权重（新增机制） |
| oldest_k_ratio | 0.12 | 最老区域选取比例 12% |
| overlap_penalty_weight | 0.0 | 探测圈重叠惩罚权重。设为 0 表示不惩罚重叠，让模型自己学习最优分工 |
| same_type_separation_weight | 0.0 | 同类分离惩罚权重。设为 0 禁用，让模型自由探索队形 |
| same_type_min_dist_ratio | 0.40 | 同类型最小距离比例（暂未启用） |
| save_dir | /home/lmx/code/RL_Benchmark/coverage_vmas/models | 模型保存路径 |

##### 关键设计决策

1. **关闭重叠/分离惩罚**：`overlap_penalty_weight=0.0` 和 `same_type_separation_weight=0.0`
   - 原因：在随机区域训练时，固定惩罚可能限制模型的适应性
   - 效果：让模型自己学习何时分散、何时集中，更灵活地适应不同大小区域

2. **提高最久未覆盖惩罚**：`max_age_penalty=0.5`（早期版本为 0.2）
   - 原因：增强对"漏检"区域的惩罚，确保模型定期巡逻所有区域

3. **新增最老区域修复奖励**：`oldest_repair_weight=0.35`
   - 目的：优先鼓励 Agent 修复当前最老的 k% 网格，避免局部区域长期荒废

4. **随机区域训练**：`randomize_area` + `coverage_margin=0.9`（默认）
   - 效果：每次 reset 时随机生成区域大小，但保证总探测能力足以覆盖（margin=90%）
   - 优势：显著提高模型泛化能力，适应不同海域大小

### 继续训练（断点续训）

从最佳模型恢复训练：
```bash
python train_coverage_vmas_mappo.py --resume_best
```

从指定模型恢复：
```bash
python train_coverage_vmas_mappo.py --resume_policy models/coverage_vmas_policy.pth --resume_critic models/coverage_vmas_critic.pth
```

### 可视化（渐变色模式）

```bash
python visualize_coverage_vmas.py --model_path models/coverage_vmas_policy_best.pth --out_dir videos --episodes 5
```

渐变色渲染效果：
- **绿色**：新鲜（刚被探测）
- **黄色**：中等（接近过期）
- **红色**：过期（需要重访）

### 可视化（二值化模式）

```bash
python visualize_coverage_vmas_binary.py --model_path models/coverage_vmas_policy_best.pth --out_dir videos --episodes 1
```

二值化渲染效果：
- **绿色半透明**：已覆盖（在重访限制内）
- **红色半透明**：未覆盖（已过期）

## 参数说明

### 训练参数

```bash
python train_coverage_vmas_mappo.py \
  --width 103.0 --height 103.0 \
  --grid_w 10 --grid_h 10 \
  --revisit_limit 10 \
  --max_steps 200 \
  --num_envs 60 \
  --n_iters 1500 \
  --lr 1e-5
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| width | 103.0 | 区域宽度（仿真单位） |
| height | 103.0 | 区域高度（仿真单位） |
| grid_w | 10 | X方向网格数 |
| grid_h | 10 | Y方向网格数 |
| revisit_limit | 10 | 重访限制步数 |
| max_steps | 200 | 每回合最大步数 |
| num_envs | 60 | 并行环境数 |
| n_iters | 1500 | 训练迭代次数 |
| lr | 1e-5 | 学习率 |
| frames_per_batch | 6000 | 每批次收集的帧数 |
| minibatch_size | 400 | 小批量大小 |
| num_epochs | 30 | 每次更新 epoch 数 |

### Agent 物理参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| speed_type_a_knots | 30.0 | A型艇速度（节） |
| speed_type_b_knots | 35.0 | B型艇速度（节） |
| sensor_range_type_a | 27.0 | A型艇探测半径（仿真单位，约25km） |
| sensor_range_type_b | 27.0 | B型艇探测半径（仿真单位，约25km） |

### 奖励函数参数（v2025.03.08+）

```python
reward = fresh_ratio                              # 基础新鲜度
         + reward_improve_weight * improve       # 改进奖励
         - max_age_penalty * max_age_norm        # 最久未覆盖惩罚
         + oldest_repair_weight * oldest_repair   # 最老区域修复奖励
         - overlap_penalty_weight * overlap       # 探测圈重叠惩罚
         - same_type_separation_weight * same_type  # 同类分离惩罚
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| reward_improve_weight | 0.5 | 改进奖励权重（鼓励持续改善覆盖） |
| max_age_penalty | 0.5 | 最久未覆盖惩罚权重（强制定期巡逻） |
| oldest_repair_weight | 0.35 | 最老区域修复奖励权重（优先修复最老网格） |
| oldest_k_ratio | 0.12 | 最老区域选取比例（选取最老的12%网格） |
| overlap_penalty_weight | 0.10 | 探测圈重叠惩罚权重 |
| same_type_separation_weight | 0.12 | 同类型平台分离惩罚权重 |
| same_type_min_dist_ratio | 0.40 | 同类型最小距离比例 |
| age_curve_power | 3.0 | 新鲜度衰减曲线幂次（非线性惩罚） |

> **最新版本更新**：相比早期版本，当前版本提高了最久未覆盖惩罚强度（0.2→0.5），并新增"最老区域修复奖励"机制，优先鼓励 Agent 修复当前最老的那批网格。

### 随机区域增强（可选）

```bash
python train_coverage_vmas_mappo.py \
  --randomize_area \
  --width_min 95 --width_max 110 \
  --height_min 95 --height_max 110 \
  --coverage_margin 0.9
```

增加环境多样性，提高模型泛化能力。`coverage_margin` 参数控制可行性判定（0.9 表示总探测能力需达到需求面积的 90%）。

## 核心算法设计

### MAPPO 架构

| 组件 | 配置 |
|------|------|
| 算法 | Multi-Agent PPO (CTDE) |
| Actor | 共享策略，参数共享 |
| Critic | 集中式 Critic（可访问全局状态） |
| 动作空间 | 连续动作 (TanhNormal) |
| 网络结构 | 2层 MLP，每层 256 单元，Tanh 激活 |

### 观测空间设计

每个 Agent 的观测向量包含：

1. **自身状态**（4维）：
   - 归一化位置 (x, y)
   - 归一化速度 (vx, vy)

2. **类型特征**（4维）：
   - 类型 one-hot 编码
   - 相对最大速度、相对最大探测范围

3. **全局特征**（3维）：
   - 全局新鲜度比例
   - 归一化最老年龄
   - 归一化步数进度

4. **局部覆盖提示**（1维）：
   - 以自身为中心的 3×3 邻域平滑新鲜度均值

5. **队友信息**（12维）：
   - 相对位置、相对速度（3个队友 × 4维）

6. **全局热图**（25维）：
   - 10×10 网格 → 5×5 平均池化

**总观测维度**：4 + 4 + 3 + 1 + 12 + 25 = **49维**

### 奖励函数详解

#### 1. 基础新鲜度奖励 (fresh_ratio)
网格新鲜度采用非线性衰减：
```python
age_norm = age / revisit_limit
age_curve = clamp(age_norm, 0, 1) ** 3.0  # 三次方
fresh_score = 1 - age_curve
```
**效果**：接近阈值时惩罚急剧增加，强制 Agent 定期巡逻。

#### 2. 改进奖励 (improve)
```python
improve = current_fresh_ratio - previous_fresh_ratio
```
鼓励 Agent 持续改善覆盖状态。

#### 3. 最久未覆盖惩罚 (max_age_norm)
```python
max_age_norm = max_age / revisit_limit
penalty = max_age_norm
```
对最老网格进行额外惩罚，确保整体均衡。

#### 4. 最老区域修复奖励 (oldest_repair) - 新增
```python
# 选取最老的 k% 网格
k = num_cells * oldest_k_ratio  # 默认12%
oldest_cells = top_k(age_no_coverage, k)
# 奖励修复这些最老网格的 Agent
oldest_repair_reward = age_reduction[oldest_cells].mean()
```
**目的**：优先鼓励 Agent 去修复当前最老的那批网格，避免局部区域长期荒废。

#### 5. 探测圈重叠惩罚 (overlap)
```python
overlap = (r_i + r_j - d_ij) / (r_i + r_j)
```
惩罚多个 Agent 探测圈过度重叠，避免资源浪费。

#### 6. 同类分离惩罚 (same_type)
```python
min_dist = same_type_min_dist_ratio * (r_i + r_j)
penalty = (min_dist - d_ij) / min_dist
```
防止同类型平台（如两艘 A 型艇）距离过近，促进分工。

### 训练技巧

1. **学习率**：默认 1e-4，训练一段时间后改为 1e-5
2. **梯度裁剪**：max_grad_norm=1.0，防止梯度爆炸
3. **PPO Clip**：clip_epsilon=0.2，控制策略更新幅度
4. **GAE 参数**：gamma=0.99, lambda=0.9
5. **防炸保护**：若 loss 为非有限值，自动跳过本次更新

## 文件结构

```
coverage_vmas/
├── scenario_coverage.py              # VMAS 场景定义
├── train_coverage_vmas_mappo.py      # 训练脚本
├── visualize_coverage_vmas.py       # 可视化（渐变色）
├── visualize_coverage_vmas_binary.py # 可视化（二值化）
├── visualize_coverage_vmas_continuous.py  # 连续渲染
├── README.md                         # 本文档
├── demo_presentation_script.md      # Demo 讲稿
├── hierarchical_central_actor_design.md  # 分层设计文档（未来扩展）
├── models/                          # 模型输出目录
│   ├── coverage_vmas_policy.pth     # 最终策略模型
│   ├── coverage_vmas_critic.pth     # 最终价值模型
│   ├── coverage_vmas_policy_best.pth   # 最佳策略模型
│   └── coverage_vmas_critic_best.pth   # 最佳价值模型
├── videos/                          # 可视化输出目录
├── old_models/                      # 历史模型备份
└── videos_old/                      # 历史可视化输出
```

## 模型输出

- `models/coverage_vmas_policy.pth` / `models/coverage_vmas_critic.pth`（最终）
- `models/coverage_vmas_policy_best.pth` / `models/coverage_vmas_critic_best.pth`（最佳）

## 进阶主题

### 与真实 DJ 任务的对齐

本实现可作为 **"离线规划器/滚动规划器"**：
- 每 10 分钟滚动生成未来一段时间的区域划分 + 任务调度 + 航线框架
- 下发给平台自动驾驶执行
- 强通信约束/失联情况下，可保留低层自治

### 方案A扩展（分层设计）

当前实现为方案B（去中心化执行 + 集中Critic）。如需更贴近"指挥中枢"模式，可参考 `hierarchical_central_actor_design.md` 文档实现：
- **中央高层Actor**：任务规划/调度（分钟级）
- **低层执行器**：运动控制/执行（秒级）

### 性能指标

训练过程中关注：
- `episode_reward_mean`：回合平均奖励
- `best_score`：历史最佳奖励
- 覆盖新鲜度比例（越高越好）
- 最大未覆盖时间（越低越好）
