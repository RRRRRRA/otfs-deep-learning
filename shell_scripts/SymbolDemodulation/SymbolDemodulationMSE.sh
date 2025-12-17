#!/bin/bash
# 声明脚本使用bash shell执行

# Slurm作业调度参数配置（告知集群管理系统如何分配资源）
#SBATCH --job-name=SymbolDemodulation  # 作业名称，用于集群中识别该任务
#SBATCH --mail-type=ALL                # 邮件通知触发条件：作业开始、结束、失败等所有状态变化
#SBATCH --mail-user=jturley1@umbc.edu  # 接收通知的邮箱地址
#SBATCH --mem=32000                    # 为作业分配32000MB（32GB）内存
#SBATCH --gres=gpu:1                   # 请求1个GPU资源（用于加速模型训练）
#SBATCH --time=24:00:00                # 作业最长运行时间限制：24小时（超时会被强制终止）
#SBATCH --error=SymbolDemodulation.err # 错误日志输出文件（记录运行中的错误信息）
#SBATCH --output=SymbolDemodulation.out # 标准输出日志文件（记录训练过程中的打印信息）

# 环境配置：加载并初始化Python运行环境
module load Anaconda3/2024.02-1        # 加载集群上的Anaconda3模块（管理Python环境的工具）
eval "$(conda shell.bash hook)"        # 初始化conda在当前shell中的环境（确保conda命令可正常使用）

conda activate 675                     # 激活名为"675"的conda虚拟环境（该环境中已安装PyTorch等依赖库）

# 执行目标Python脚本：启动基于MSE损失的OTFS符号解调模型训练/推理
python ~/gokhale_user/675/project/python_scripts/SymbolDemodulationMSE.py
