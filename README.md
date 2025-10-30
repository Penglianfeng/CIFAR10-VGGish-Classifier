# CIFAR10-VGGish-Classifier

简介  
本仓库包含一个交互式 Jupyter Notebook（the-innovation-class-assignment-is-about-the-class.ipynb），演示如何使用一个受 VGG16 启发的轻量 CNN 在 CIFAR-10 数据集上进行训练与评估。Notebook 覆盖数据加载、标准化、one-hot 编码、实时数据增强（ImageDataGenerator）、模型构建（Conv2D + BatchNormalization + Dropout + L2 正则化）、训练回调（ReduceLROnPlateau、EarlyStopping）以及训练过程的可视化与最终评估。

主要特点
- 使用 Keras / TensorFlow 加载 CIFAR-10 数据集并切分训练/验证/测试集
- 使用 ImageDataGenerator 做实时数据增强：rotation、shift、flip、zoom、brightness、shear、channel shift 等
- 构建一个 VGG-like 的卷积神经网络（32→64→128→256 filters），使用 BatchNormalization、Dropout 与 L2 正则化
- 训练配置：Adam(learning_rate=0.0005)，batch_size=64，epochs=300，ReduceLROnPlateau (factor=0.5, patience=10)，EarlyStopping (patience=40, restore_best_weights=True)
- 可视化训练/验证的 loss 与 accuracy，并在测试集上评估最终性能

文件结构（仓库示例）
- the-innovation-class-assignment-is-about-the-class.ipynb — 主 Notebook（包含全部训练与可视化单元）
- README.md — 项目说明（本文件）
- LICENSE — GPL-2.0 许可证
- .gitignore

运行环境与依赖
- 推荐 Python 版本：3.8 - 3.11
- 主要依赖（示例）：tensorflow (包含 Keras API), numpy, matplotlib, opencv-python, scikit-learn  
更多依赖请参见 requirements.txt（若仓库中未包含，请创建）。

快速上手
1. 克隆仓库：
```bash
git clone https://github.com/Penglianfeng/StudentSystem.git
cd StudentSystem
```

2. 创建并激活虚拟环境（可选）：
```bash
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
.venv\Scripts\activate      # Windows
```

3. 安装依赖（如果你已经有 requirements.txt）：
```bash
pip install -r requirements.txt
```
若没有 requirements.txt：
```bash
pip install tensorflow matplotlib numpy opencv-python scikit-learn
```

4. 启动并运行 Notebook：
```bash
jupyter notebook the-innovation-class-assignment-is-about-the-class.ipynb
```
按单元顺序运行 Notebook，或者在命令行将 Notebook 转为脚本后运行（见下方建议）。

关键实现细节（基于 Notebook 的代码）
- 数据处理：
  - 使用 cifar10.load_data() 直接获取 (X_train, y_train), (X_test, y_test)
  - train/valid 划分：train_test_split(..., test_size=0.1, random_state=0)
  - 统一将数据转为 float32 并按训练集均值与标准差归一化：X = (X - mean) / (std + 1e-7)
  - 标签使用 to_categorical(one-hot) 变换为 10 类
- 数据增强（ImageDataGenerator 参数）：
  - rotation_range=15, width_shift_range=0.12, height_shift_range=0.12, horizontal_flip=True
  - zoom_range=0.1, brightness_range=[0.9,1.1], shear_range=10, channel_shift_range=0.1
- 模型结构（概要）：
  - Conv2D(32) + BN, Conv2D(32) + BN -> MaxPool -> Dropout(0.2)
  - Conv2D(64) x2 + BN -> MaxPool -> Dropout(0.3)
  - Conv2D(128) x2 + BN -> MaxPool -> Dropout(0.4)
  - Conv2D(256) x2 + BN -> MaxPool -> Dropout(0.5)
  - Flatten -> Dense(10, softmax)
  - L2 正则化（weight_decay=1e-4）在 Conv 层中使用
- 训练设置：
  - optimizer = Adam(learning_rate=0.0005)
  - loss = categorical_crossentropy, metrics=['accuracy']
  - callbacks: ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-5) 与 EarlyStopping(monitor='val_loss', patience=40, restore_best_weights=True)
  - batch_size=64, epochs=300（EarlyStopping 会提前停止）
- 可视化：
  - 绘制训练/验证 loss 和 accuracy 曲线
  - 最终 evaluate(X_test, y_test) 打印 test accuracy 和 test loss

运行建议与注意事项
- GPU：建议在带 GPU 的环境运行（如 CUDA + cuDNN 支持），训练速度会显著提升。Notebook 内曾检测到 Tesla P100 的消息。
- 文件体积与 Notebook 输出：如果 Notebook 当前包含大量嵌入输出（图像），建议清理输出后再提交以便版本控制（jupyter nbconvert --clear-output --inplace …）。
- 隐私：确认 Notebook 中没有敏感信息（API keys、个人学号等）。
- 可重复性：建议将依赖锁定到 requirements.txt，并在 README 中注明 Python / TF 版本。
- 可扩展改进方向：引入残差块（ResNet）、更强的数据增强（AutoAugment / RandAugment）、学习率调度（CosineAnnealing）、混合精度训练、模型裁剪或蒸馏以压缩模型。

示例 requirements.txt（建议内容）
```text name=requirements.txt
tensorflow>=2.10
numpy>=1.23
matplotlib>=3.5
opencv-python
scikit-learn
```

贡献
欢迎提交 Issue 或 Pull Request。你可以：
- 清理 Notebook 输出并提交干净版本
- 添加 requirements.txt 与 environment.yml（用于 conda）
- 提供训练好的 model.h5 或导出为 SavedModel（注意文件体积）

许可证
本项目使用 LICENSE 中的 GNU GPL-2.0。

联系方式
在仓库中打开 Issue 或通过 GitHub 个人资料联系仓库维护者。
