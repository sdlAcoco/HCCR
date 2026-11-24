# HCCR

## 安装

安装 Git 和 Git LFS 后，克隆仓库并下载权重。

```sh
git lfs install
git clone https://github.com/sdlAcoco/HCCR.git
cd HCCR
git lfs pull
```

在 Python 3.9 环境中安装依赖。后续命令均在仓库根目录执行。

```sh
python -m pip install torch==2.1.2 numpy==1.26.4 pandas==2.2.3 scipy==1.13.1 scikit-learn==1.6.1
```

## 训练

下文中的 `SERIES` 是占位符。运行 `python -m hccr fit --help` 查看 `--series` 支持的名称，再将目录名、文件名前缀和命令中的 `SERIES` 替换为同一个名称。

```text
data/
└── SERIES/
    ├── SERIES_train_data.npy
    ├── SERIES_train_date.npy
    ├── SERIES_test_data.npy
    ├── SERIES_test_date.npy
    └── SERIES_test_label.npy
```

`*_data.npy` 按“时间点 × 通道”存放观测值，`*_date.npy` 存放逐行对应的时间戳。`*_test_label.npy` 为每个测试时间点提供一个标签，0 表示正常，1 表示异常。

```sh
python -m hccr fit --series SERIES --data-root ./data --save-to ./runs
```

程序按验证损失选择最佳权重，首次训练的权重保存在 `./runs/SERIES/run-01/best.pt`。可用 `--rounds` 设置最大训练轮数，用 `--batch` 设置批大小。

## 测试

**测试自己训练的权重**时，沿用训练时的名称和数据目录，通过 `--weights` 指定权重文件。

```sh
python -m hccr score --series SERIES --data-root ./data --weights ./runs/SERIES/run-01/best.pt --report ./runs/test.json
```

**测试仓库自带的权重**时，准备与该权重匹配的三个测试文件（观测值、时间戳和标签），保留 `weights/` 中配套的权重和配置文件，使用默认参数运行。

```sh
python -m hccr score --data-root ./data --report ./runs/test.json
```

测试报告写入 `./runs/test.json`。程序会在 CUDA 可用时使用 GPU，也可以在训练或测试命令末尾添加 `--device cpu` 指定 CPU。
