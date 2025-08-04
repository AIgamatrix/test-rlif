# test-rlif

这是一个集成了多个强化学习和数学推理项目的对比测试仓库。

## 项目结构

### 主要项目

- **EMPO/**: Right question is already half the answer: Fully unsupervised llm reasoning incentivization
  - 包含数学评估工具和自然语言评估工具
  - 支持TRL和VERL两种训练框架

- **Intuitor/**: Learning to Reason without External Rewards
  - 数学直觉推理模型训练
  - 包含open-r1-intuitor和verl-intuitor两个版本

- **rent-rl/**: RENT: Reinforcement Learning via Entropy Minimization
  - 基于VERL框架的PPO训练
  - 支持多种算法配置

- **TTRL/**: Ttrl: Test-time reinforcement learning
  - 测试时强化学习框架

### 训练脚本

- `empo-train.sh`: EMPO项目训练脚本
- `intuitor-train.sh`: Intuitor项目训练脚本
- `rent-train.sh`: rent-rl项目训练脚本
- `ttrl-train.sh`: TTRL项目训练脚本

### 配置文件

- `set_path.sh`: 环境路径设置脚本
- `requirement_*.txt`: 各项目的依赖包列表

## 使用方法
**确保python版本为3.10**
**cuda版本为12**

1. 首先运行环境设置脚本：
   ```bash
   source ./set_path.sh
   ```

2. 安装对应项目的依赖：
   ```bash
   pip install -r requirement_<project_name>.txt
   ```

3. 运行对应的训练脚本：
   ```bash
   bash <project_name>-train.sh
   ```

## 注意事项

- 至少80GB显存
