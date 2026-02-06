# VMAS 覆盖任务 MAPPO（方案A）

使用 VMAS 自定义场景实现“矩形区域覆盖 + 重访约束”，两种 agent 类型（各 2 个），差异仅为**速度**与**探测范围**。

## 运行

```bash
python train_coverage_vmas_mappo.py
```

如果希望每次训练随机矩形大小，并保证“可扫完”的可行性：

```bash
python train_coverage_vmas_mappo.py --randomize_area --width_min 0.8 --width_max 1.2 --height_min 0.8 --height_max 1.2 --coverage_margin 0.9
```

## 可视化（MP4）

```bash
python visualize_coverage_vmas.py --model_path models/coverage_vmas_policy_best.pth --out_dir videos --episodes 1
```

说明：可视化会用绿色半透明标出“10分钟内已覆盖”的网格区域。

## 参数说明

```bash
python train_coverage_vmas_mappo.py \
  --width 1.0 --height 1.0 \
  --grid_w 10 --grid_h 10 \
  --revisit_limit 10 \
  --max_steps 200 \
  --num_envs 60
```

模型输出：
- `models/coverage_vmas_policy.pth` / `models/coverage_vmas_critic.pth`（最终）
- `models/coverage_vmas_policy_best.pth` / `models/coverage_vmas_critic_best.pth`（最佳）
