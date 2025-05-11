# Pixel-level-Object-Operating-System-

该项目来自作者本人的本科生毕业设计。
这是一个基于Python的抓取项目，使用Flask作为Web框架。该项目集成了PointNet++和DGCNN模型用于点云处理。
需要放在训练的环境下运行，确保models文件夹里包含对应的网络。

## 环境要求
- Python 3.7+
- Conda环境管理工具
- CUDA支持（用于PyTorch GPU加速）

## 运行项目

1. 确保已激活正确的Conda环境：
```bash
conda activate ICAF-4
```

2. 进入项目目录：
```bash
cd /home/guoyuefan/grasp
```

3. 启动主程序：
```bash
python main.py
```

## 项目结构

- `main.py`: 主程序入口
- `app.py`: Flask应用配置
- `grasp_vis.py`: 核心脚本

## 依赖包

### 基础依赖
- Flask >= 2.0.1
- Flask-CORS >= 3.0.10
- NumPy >= 1.19.5
- h5py >= 3.1.0
- PyTorch >= 1.7.1
- SciPy >= 1.6.0

### PointNet++和DGCNN相关依赖
- numpy
- msgpack-numpy
- lmdb
- h5py
- hydra-core==0.11.3
- pytorch-lightning==0.7.1

## 注意事项

- 请确保在运行项目前已正确安装所有依赖包
- 确保使用正确的Python版本（建议使用Python 3.7或更高版本）
- 如果遇到权限问题，请确保有适当的文件访问权限
- 安装PointNet++自定义操作时可能需要CUDA支持
- 确保系统已安装CUDA和对应版本的cuDNN（如果使用GPU加速）

## 常见问题

如果遇到问题，请检查：
1. Conda环境是否正确激活
2. 所有依赖包是否已正确安装
3. 项目路径是否正确
4. PointNet++自定义操作是否成功编译
5. CUDA环境是否正确配置（如果使用GPU）

