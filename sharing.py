"""模 257 的 Shamir 与 Brickell 图像秘密共享。"""
from pathlib import Path
import secrets
import numpy as np

P = 257


def check(k, n):
    if type(k) is not int or type(n) is not int or not 2 <= k <= n <= 256:
        raise ValueError("参数必须满足 2≤k≤n≤256")


def vectors(k, n):
    check(k, n)
    return np.array([[pow(x, j, P) for j in range(k)] for x in range(1, n + 1)], dtype=np.int64)


def random_field(shape):
    # 拒绝 65535，使每个域元素恰有 255 个原像，避免取模偏差。
    out = np.empty(shape, dtype=np.int64)
    flat, done = out.reshape(-1), 0
    while done < flat.size:
        raw = np.frombuffer(secrets.token_bytes(2 * (flat.size - done)), dtype="<u2")
        valid = raw[raw < 65535]
        flat[done:done + valid.size] = valid % P
        done += valid.size
    return out


def solve_mod(matrix, rhs):
    """有限域高斯消元；欠定系统将自由变量置零，无解时拒绝。"""
    a = np.column_stack((matrix, rhs)).astype(np.int64) % P
    rows, cols = len(a), a.shape[1] - 1
    pivots, row = [], 0
    for col in range(cols):
        candidates = np.flatnonzero(a[row:, col])
        if not len(candidates):
            continue
        pivot = row + int(candidates[0])
        a[[row, pivot]] = a[[pivot, row]]
        a[row] = a[row] * pow(int(a[row, col]), -1, P) % P
        for other in range(rows):
            if other != row:
                a[other] = (a[other] - a[other, col] * a[row]) % P
        pivots.append(col)
        row += 1
        if row == rows:
            break
    if np.any((a[:, :cols] == 0).all(axis=1) & (a[:, -1] != 0)):
        raise ValueError("有限域方程无解")
    solution = np.zeros(cols, dtype=np.int64)
    for row, col in enumerate(pivots):
        solution[col] = a[row, -1]
    return solution


def selected_ids(k, n, ids):
    check(k, n)
    ids = tuple(ids)
    if len(ids) < k or len(set(ids)) != len(ids) or any(type(i) is not int or not 1 <= i <= n for i in ids):
        raise ValueError("至少需要 k 个有效且不重复的份额编号")
    return ids[:k]


def weights(method, k, n, ids, public=None):
    selected = selected_ids(k, n, ids)
    if method == "shamir":
        result = []
        for x in selected:
            numerator, denominator = 1, 1
            for y in selected:
                if x != y:
                    numerator = numerator * (-y) % P
                    denominator = denominator * (x - y) % P
            result.append(numerator * pow(denominator, -1, P) % P)
        return np.array(result, dtype=np.int64)
    if method == "brickell":
        public = vectors(k, n) if public is None else public
        target = np.eye(1, k, dtype=np.int64)[0]
        return solve_mod(public[np.array(selected) - 1].T, target)
    raise ValueError("未知算法")


def split(image, method, k, n, public=None):
    check(k, n)
    if image.dtype != np.uint8 or image.ndim not in (2, 3) or image.size == 0:
        raise ValueError("图像必须是非空的 uint8 灰度或多通道数组")
    # 每个像素及通道独立采样；最高次系数允许为零。
    coeff = np.vstack((image.reshape(1, -1).astype(np.int64), random_field((k - 1, image.size))))
    if method == "shamir":
        shares = np.empty((n, image.size), dtype=np.uint16)
        for i in range(n):
            value = np.zeros(image.size, dtype=np.int64)
            for row in coeff[::-1]:
                value = (value * (i + 1) + row) % P
            shares[i] = value
    elif method == "brickell":
        public = vectors(k, n) if public is None else public
        shares = ((public @ coeff) % P).astype(np.uint16)
    else:
        raise ValueError("未知算法")
    return shares.reshape((n,) + image.shape)


def recover(shares, ids, method, k, n, precomputed=None):
    ids = tuple(ids)
    selected = selected_ids(k, n, ids)
    if shares.dtype != np.uint16 or len(shares) != len(ids) or np.any(shares >= P):
        raise ValueError("份额类型、编号数量或域取值不合法")
    if precomputed is None:
        chosen = weights(method, k, n, ids)
    else:
        saved_method, saved_ids, chosen = precomputed
        if saved_method != method or tuple(saved_ids) != selected or len(chosen) != k:
            raise ValueError("预计算权重与算法、份额编号不匹配")
    value = np.zeros(shares.shape[1:], dtype=np.int64)
    for weight, share in zip(chosen, shares[:k]):
        value = (value + int(weight) * share.astype(np.int64)) % P
    if np.any(value > 255):
        raise ValueError("恢复值超出像素范围，份额可能损坏")
    return value.astype(np.uint8)


def save_shares(folder, shares, method, k, n, mode):
    """保存完整 16 位份额及公开元数据，不保存随机系数或原图摘要。"""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    session = secrets.token_hex(16)
    paths = []
    for i, share in enumerate(shares, 1):
        path = folder / f"share_{i}.npz"
        np.savez(path, data=share, id=i, k=k, n=n, p=P, method=method, mode=mode, session=session)
        paths.append(path)
    return paths


def load_shares(paths):
    arrays, ids, common = [], [], None
    for path in paths:
        with np.load(path, allow_pickle=False) as file:
            metadata = {key: file[key].item() for key in ("k", "n", "p", "method", "mode", "session")}
            data, index = file["data"].copy(), int(file["id"])
        if common is not None and (metadata != common or data.shape != arrays[0].shape):
            raise ValueError("份额来自不同会话、参数或图像形状")
        if data.dtype != np.uint16 or np.any(data >= P) or metadata["p"] != P:
            raise ValueError("份额数据不合法")
        common = metadata
        arrays.append(data)
        ids.append(index)
    if not arrays:
        raise ValueError("未提供份额")
    check(common["k"], common["n"])
    if len(set(ids)) != len(ids) or any(not 1 <= i <= common["n"] for i in ids):
        raise ValueError("份额编号重复或越界")
    return np.stack(arrays), ids, common
