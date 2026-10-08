"""生成校徽份额、验证门限，并重复测量时间与可跟踪内存峰值。"""
import argparse
import csv
import gc
import itertools
import json
import platform
import time
import tracemalloc
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import sharing as ss


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def thumbnail(array, mode, size=160):
    image = Image.fromarray(array)
    if mode == "RGBA":
        background = Image.new("RGBA", image.size, "white")
        image = Image.alpha_composite(background, image).convert("RGB")
    else:
        image = image.convert("RGB")
    image.thumbnail((size, size))
    return image


def make_preview(image, shares, restored, mode, path):
    # 预览采用 RGB；透明通道仍保存在真实份额中，不用预览恢复。
    items = [("Original", image), ("Recovered", restored)]
    for i, share in enumerate(shares, 1):
        visible = ((share.astype(np.uint32) * 255) // 256).astype(np.uint8)
        if mode == "RGBA":
            visible = visible[..., :3]
        items.append((f"Share {i}", visible))
    cols, cell = 4, 180
    rows = (len(items) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * cell, rows * (cell + 24)), "white")
    draw = ImageDraw.Draw(canvas)
    for j, (label, array) in enumerate(items):
        x, y = (j % cols) * cell, (j // cols) * (cell + 24)
        draw.text((x + 10, y + 5), label, fill="black")
        picture = thumbnail(array, mode if label in ("Original", "Recovered") else "RGB")
        canvas.paste(picture, (x + 10, y + 28))
    canvas.save(path)


def verify(image, shares, method, k, n):
    public, results = ss.vectors(k, n), []
    for count in range(k, n + 1):
        for ids in itertools.combinations(range(1, n + 1), count):
            restored = ss.recover(shares[np.array(ids) - 1], ids, method, k, n)
            mismatch = int(np.count_nonzero(restored != image))
            results.append({"ids": ids, "equal": mismatch == 0, "mismatched_values": mismatch})
            if mismatch:
                raise AssertionError("恢复图像不一致")
    unauthorized = 0
    target = np.eye(1, k, dtype=np.int64)[0]
    for count in range(k):
        for ids in itertools.combinations(range(1, n + 1), count):
            try:
                ss.solve_mod(public[np.array(ids, dtype=int) - 1].T, target)
            except ValueError:
                unauthorized += 1
            else:
                raise AssertionError("不足门限的向量能够恢复秘密")
            try:
                ss.recover(shares[np.array(ids, dtype=int) - 1], ids, method, k, n)
            except ValueError:
                pass
            else:
                raise AssertionError("恢复接口未拒绝不足门限的输入")
    # 用同一组已保存份额的一个像素，构造两个不同且合法的秘密解释。
    ids = tuple(range(1, k))
    matrix = np.vstack((public[:k - 1], target))
    known = shares[:k - 1].reshape(k - 1, -1)[:, 0].astype(np.int64)
    secret = int(image.reshape(-1)[0])
    alternatives = [secret, (secret + 1) % 256]
    for candidate in alternatives:
        coeff = ss.solve_mod(matrix, np.append(known, candidate))
        assert np.array_equal(public[:k - 1] @ coeff % ss.P, known)
        assert int(coeff[0]) == candidate
    return {"authorized": results, "authorized_combinations": len(results),
            "unauthorized_combinations": unauthorized, "all_unauthorized_rejected": True,
            "target_not_in_unauthorized_span": True,
            "ambiguity_witness": {"ids": ids, "candidate_secrets": alternatives,
                                  "both_match_observed_shares": True}}


def measured(image, method, k, n, ids):
    gc.collect()
    tracemalloc.start()
    secret = image.copy()
    peaks, record = [tracemalloc.get_traced_memory()[1]], {}

    def stage(name, operation):
        tracemalloc.reset_peak()
        start = time.perf_counter_ns()
        value = operation()
        record[name + "_ms"] = (time.perf_counter_ns() - start) / 1e6
        peak = tracemalloc.get_traced_memory()[1]
        record[name + "_peak_bytes"] = peak
        peaks.append(peak)
        return value

    try:
        def initialize():
            public = ss.vectors(k, n) if method == "brickell" else None
            prepared = (method, tuple(ids), ss.weights(method, k, n, ids, public))
            return public, prepared
        public, prepared = stage("init", initialize)
        shares = stage("split", lambda: ss.split(secret, method, k, n, public))
        restored = stage("recover", lambda: ss.recover(
            shares[np.array(ids) - 1], ids, method, k, n, prepared))
        record["whole_peak_bytes"] = max(peaks)
        record["equal"] = bool(np.array_equal(secret, restored))
        record["share_array_bytes"] = shares.nbytes
        if not record["equal"]:
            raise AssertionError("性能测量恢复失败")
        return record
    finally:
        tracemalloc.stop()


def main():
    parser = argparse.ArgumentParser(description="任务一：图像秘密共享")
    parser.add_argument("--image", default="assets/logo.png")
    parser.add_argument("--output", default="results")
    parser.add_argument("--params", nargs="+", default=["3,5", "4,6"])
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    try:
        params = [tuple(map(int, text.split(","))) for text in args.params]
    except ValueError:
        parser.error("参数格式应为 k,n")
    if any(len(pair) != 2 for pair in params):
        parser.error("参数格式应为 k,n")
    if len(params) < 2 or len(set(params)) < 2 or args.repeats < 5:
        parser.error("至少两组不同参数，每组重复至少 5 次")
    for k, n in params:
        ss.check(k, n)
        if n > 10:
            parser.error("穷举实验将 n 限制为 10；数学核心支持 n≤256")
    with Image.open(args.image) as source:
        mode = source.mode if source.mode in ("L", "RGB", "RGBA") else "RGBA"
        image = np.asarray(source.convert(mode), dtype=np.uint8).copy()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(out / "original.png")
    groups, records = [], []
    for k, n in params:
        ids = tuple(range(1, k + 1))
        for method in ("shamir", "brickell"):
            folder = out / f"{method}_{k}_{n}"
            folder.mkdir(exist_ok=True)
            shares = ss.split(image, method, k, n)
            paths = ss.save_shares(folder / "shares", shares, method, k, n, mode)
            restored_shares, loaded_ids, _ = ss.load_shares(paths[:k])
            restored = ss.recover(restored_shares, loaded_ids, method, k, n)
            assert np.array_equal(restored, image)
            Image.fromarray(restored).save(folder / "recovered.png")
            make_preview(image, shares, restored, mode, folder / "preview.png")
            correctness = verify(image, shares, method, k, n)
            write_json(folder / "verification.json", correctness)
            groups.append({"method": method, "k": k, "n": n,
                           "authorized_combinations": correctness["authorized_combinations"],
                           "unauthorized_combinations": correctness["unauthorized_combinations"],
                           "disk_roundtrip_equal": True,
                           "share_array_bytes": shares.nbytes,
                           "share_file_bytes": sum(p.stat().st_size for p in paths)})
            # 同一恢复组合预热一次，文件读写与预览均不进入计时。
            measured(image, method, k, n, ids)
        for repeat in range(args.repeats):
            order = ("shamir", "brickell") if repeat % 2 == 0 else ("brickell", "shamir")
            for method in order:
                row = {"method": method, "k": k, "n": n, "repeat": repeat,
                       "recovery_ids": ids, **measured(image, method, k, n, ids)}
                records.append(row)
        print(f"已完成参数 ({k},{n}) 的图像验证和 {args.repeats} 次配对测量", flush=True)
    summaries = []
    metrics = ["init_ms", "split_ms", "recover_ms", "init_peak_bytes", "split_peak_bytes",
               "recover_peak_bytes", "whole_peak_bytes"]
    for group in groups:
        rows = [r for r in records if all(r[key] == group[key] for key in ("method", "k", "n"))]
        summaries.append({**group, "repeats": len(rows),
                          **{name + "_mean": float(np.mean([r[name] for r in rows])) for name in metrics},
                          **{name + "_max": max(r[name] for r in rows) for name in metrics if "peak" in name}})
    result = {"environment": {"os": platform.platform(), "python": platform.python_version(),
                               "numpy": np.__version__, "pillow": Image.__version__,
                               "processor": platform.processor() or "未读取处理器型号"},
              "image": {"shape": image.shape, "mode": mode, "pixel_values": image.size},
              "field": ss.P, "summary": summaries, "records": records,
              "measurement": {"memory": "tracemalloc 可跟踪分配，已验证包含 NumPy 数据缓冲；不是进程 RSS。阶段峰值含当时常驻输入、份额及辅助数组，全过程峰值为各阶段与输入拷贝峰值的最大值。排除原始输入读取与解释器/库常驻内存。",
                              "timing": "时间与内存同步测量，含 tracemalloc 开销；分片包含独立密码随机数采样；初始化含公开矩阵和权重预计算；恢复使用预计算权重；不含文件读写和预览。",
                              "comparison": "两方案同图、同域、同参数、同恢复编号，均 NumPy 向量化，交替执行，每组各预热一次。"}}
    write_json(out / "benchmark.json", result)
    with (out / "benchmark.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print("全部组合像素一致；不足门限已验证无法唯一确定秘密。")


if __name__ == "__main__":
    main()
