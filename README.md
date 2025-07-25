# Se_Mu_AD: Second-Order Multi-Modal Fusion and Dynamic Optimization for Robust Multi-Task Autonomous Driving

**Se_Mu_AD** 是一个用于自动驾驶任务的结构化多任务学习框架，支持任务亲和度估计、动态分组和CARLA环境下的端到端控制策略训练与评估。该框架整合了轨迹生成等模块，适用于CARLA模拟器与自动驾驶研究中的多任务优化场景。

## 🔧 环境配置

请使用 `conda` 创建项目所需的虚拟环境：

```bash
conda env create -f environment.yml
conda activate semuad

此外，请安装 CARLA 和其依赖：
bash setup_carla.sh

## 训练与测试
1. 生成训练数据
python generate_dataset_slurm.py

2. 执行模型评估
python evaluate_routes_slurm.py


## 项目结构
Se_Mu_AD/
│
├── team_code/                  # 多任务策略与模型主干实现
├── leaderboard/                # CARLA Leaderboard接口适配模块
├── scenario_runner/            # 情景脚本执行模块
├── tools/                      # 实验工具与辅助脚本
├── docs/                       # 文档与可视化支持
├── assets/                     # 配置与示例数据
│
├── generate_dataset_slurm.py  # 数据生成脚本（支持SLURM调度）
├── evaluate_routes_slurm.py   # 评估脚本
├── evaluate_routes_slurm_dw.py# 多样性扩展评估脚本
├── environment.yml            # Conda环境配置
├── setup_carla.sh             # CARLA自动安装脚本
├── LICENSE                    # 开源协议（默认为MIT）

本项目遵循 MIT License

