"""Top-down picture of plan.json: lane centre lines, crosswalks, actor paths, actor positions every 4 s
(optionally over a top-down CARLA semantic image with its pose json). venv (Python 3.12, PIL).
usage: plot_plan.py PLAN_JSON OUT_PNG [--bg TOPDOWN_PNG TOPDOWN_JSON]"""
import json, math, sys
from PIL import Image, ImageDraw, ImageFont

plan = json.load(open(sys.argv[1]))
OUT = sys.argv[2]
CX, CY, HALF, PX = 425.0, 512.0, 75.0, 1400          # view centre, half size (m), image size (px)
sc = PX / (2 * HALF)
COL = {"ego": (220, 30, 30), "lead": (30, 90, 220), "cyclist": (0, 160, 0), "oncoming_lt": (230, 120, 0),
       "oncoming_rt": (150, 60, 200), "oncoming_stop": (120, 80, 40)}


def P(x, y):                                        # CARLA x -> image right, CARLA y -> image down (top view)
    return ((x - CX) * sc + PX / 2, (y - CY) * sc + PX / 2)


if "--bg" in sys.argv:
    i = sys.argv.index("--bg")
    img = Image.open(sys.argv[i + 1]).convert("RGB")
    bg = json.load(open(sys.argv[i + 2]))            # {"cx", "cy", "half"}: what the top-down camera covers
    cut = img.crop((0, 0, img.width, img.height)).resize((int(2 * bg["half"] * sc),) * 2)
    im = Image.new("RGB", (PX, PX), (255, 255, 255))
    x0, y0 = P(bg["cx"] - bg["half"], bg["cy"] - bg["half"])
    im.paste(Image.blend(cut, Image.new("RGB", cut.size, (255, 255, 255)), 0.45), (int(x0), int(y0)))
else:
    im = Image.new("RGB", (PX, PX), (255, 255, 255))
d = ImageDraw.Draw(im)
font = ImageFont.load_default(16) if hasattr(ImageFont, "load_default") else None
for x, y, r, l, j in plan["lanes"]:
    px, py = P(x, y)
    d.ellipse((px - 1.5, py - 1.5, px + 1.5, py + 1.5), fill=(170, 170, 170) if not j else (200, 200, 120))
for name, q in plan["meta"]["crosswalks"].items():
    d.polygon([P(p[0], p[1]) for p in q], outline=(0, 0, 0))
for a in plan["actors"]:
    c = COL.get(a["name"], (0, 0, 0) if a["kind"] == "walker" else (90, 90, 90))
    d.line([P(p[0], p[1]) for p in a["path"]], fill=c, width=1)
    for k in range(0, len(a["poses"]), 40):
        x, y, z, yaw, s = a["poses"][k]
        L, W = {"car": (4.8, 2.0), "bike": (1.8, 0.7), "walker": (0.6, 0.6)}[a["kind"]]
        cs, sn = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
        corners = [(x + cs * u - sn * v, y + sn * u + cs * v) for u, v in ((L/2, W/2), (L/2, -W/2), (-L/2, -W/2), (-L/2, W/2))]
        d.polygon([P(*p) for p in corners], outline=c, fill=None)
        px, py = P(x, y)
        d.text((px + 4, py - 8), "%d" % (k // 10), fill=c, font=font)
    x, y = a["poses"][0][:2]
    d.text((P(x, y)[0] + 6, P(x, y)[1] + 6), a["name"], fill=c, font=font)
d.text((10, 10), "positions every 4 s (labels: t in s); scale %.1f px/m" % sc, fill=(0, 0, 0), font=font)
im.save(OUT)
print("wrote", OUT)
