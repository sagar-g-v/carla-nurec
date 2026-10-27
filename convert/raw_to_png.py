"""Convert the BGRA buffers of `record.py --raw` into RGB PNGs (same pixels as CARLA's save_to_disk), in parallel,
and delete the raw files. usage: raw_to_png.py REC_DIR [--workers 12]"""
import glob, json, os, sys
from multiprocessing import Pool
import numpy as np
from PIL import Image


def conv(args):
    p, w, h = args
    a = np.fromfile(p, np.uint8).reshape(h, w, 4)
    Image.fromarray(a[..., [2, 1, 0]]).save(p[:-5] + ".png", compress_level=1)
    os.remove(p)
    return 1


if __name__ == "__main__":
    rec = sys.argv[1]
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 12
    w, h = json.load(open(os.path.join(rec, "meta.json")))["res"]
    files = glob.glob(os.path.join(rec, "**", "*.bgra"), recursive=True)
    with Pool(workers) as pool:
        n = sum(pool.imap_unordered(conv, [(f, w, h) for f in files], chunksize=16))
    print("converted", n, "images")
