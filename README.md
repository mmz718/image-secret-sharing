# 实验三 图像秘密共享

实现任务一，分别采用 Shamir 多项式与 Brickell 向量内积。校徽为 267×267 的 RGBA 图，保留透明通道，有限域为模 257，默认门限参数为 `(3,5)`、`(4,6)`。

## 运行

Python 3.11+。在本目录执行：

```powershell
python -m pip install -r requirements.txt
python test_sharing.py
python experiment.py
```

也可指定输入与测量次数：

```powershell
python experiment.py --image assets/logo.png --params 3,5 4,6 --repeats 5
```

原图、恢复 PNG、份额预览及完整 NPZ 份额位于 `results/`。完整份额包含值 256，使用 `uint16`；预览图仅供展示，不能用于恢复。

只凭指定份额恢复，不需要原图：

```powershell
python restore.py results/shamir_3_5/shares/share_1.npz results/shamir_3_5/shares/share_3.npz results/shamir_3_5/shares/share_5.npz --output recovered.png
```

## 文件

- `sharing.py`：随机采样、两种分片、模域求解、恢复、份额保存与读取。
- `experiment.py`：图像展示、组合验证、重复测量与 CSV/JSON 输出。
- `restore.py`：从完整份额恢复 PNG。
- `test_sharing.py`：17 个数学、文件与边界测试。


## 验证和统计

穷举两组参数下所有不少于 k 的集合，共 76 个合法组合，RGBA 像素全部一致；116 个不足门限组合（含空集合）均拒绝，并验证目标向量不在其张成空间。另给出实际份额对应两个合法秘密的例子。四组完整份额从磁盘读回均一致。

每组每算法预热一次、测量至少 5 次，使用相同输入、域、参数与恢复编号，交替执行。初始化单列公开矩阵与权重预计算；分片包含密码随机数；恢复使用预计算权重；文件读写和预览不计入时间。时间与内存同时测量，包含追踪开销。

`tracemalloc` 已验证覆盖 NumPy 数据缓冲。阶段峰值含当时常驻输入、份额和临时数组；全过程峰值取输入拷贝和各阶段峰值的最大值。它不是进程 RSS，排除解释器/库和外部原图数组。保存逐次峰值、均值和跨次最大值；份额总存储统计全部 NPZ，另外记录数组净字节数。

## 原理与条件

随机系数对每个像素和通道独立、均匀生成。Brickell 使用 Vandermonde 公开向量，其门限实例与 Shamir 数学等价；实现路径不同，不将向量化与临时数组差异直接归因于算法。

只保证有效份额的恢复，不提供认证或纠错。公开元数据仅含编号、门限、模数、算法、模式和会话标识；不保存随机系数或原图摘要。数学核心支持 `2≤k≤n≤256`，穷举实验将 n 限制为 10。
