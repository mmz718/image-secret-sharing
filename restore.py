"""仅凭完整份额文件恢复图像，不读取原图。"""
import argparse
from PIL import Image
from sharing import load_shares, recover


def main():
    parser = argparse.ArgumentParser(description="从至少 k 个份额恢复图像")
    parser.add_argument("shares", nargs="+")
    parser.add_argument("--output", default="recovered.png")
    args = parser.parse_args()
    shares, ids, meta = load_shares(args.shares)
    image = recover(shares, ids, meta["method"], meta["k"], meta["n"])
    Image.fromarray(image).save(args.output)
    print("已恢复图像：" + args.output)


if __name__ == "__main__":
    main()
