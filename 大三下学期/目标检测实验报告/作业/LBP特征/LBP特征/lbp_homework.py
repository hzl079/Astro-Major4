import math
import os
import struct
import tempfile
import urllib.request

# 作业默认：512×512，P=8，R=1 的圆形 LBP + 旋转不变 + 等价模式(U2)
SIZE, P, R = 512, 8, 1.0
OUT = "results"
LOCAL = "lena.png"
URL = "https://raw.githubusercontent.com/mikolalysenko/lena/master/lena.png"

OFFSETS = [(-R * math.sin(2 * math.pi * i / P), R * math.cos(2 * math.pi * i / P)) for i in range(P)]


def clamp(x):
    return max(0, min(255, x))


def rgb(s):
    if isinstance(s, tuple):
        return tuple(int(x) for x in s[:3])
    if s.startswith("#"):
        return tuple(int(s[i : i + 2], 16) for i in (1, 3, 5))
    return tuple(int(x) for x in s.split()[:3])


def fallback_color(size):
    """无网络/无 tk 时用简单彩色图，满足“彩色输入”即可。"""
    c = size / 2
    img = []
    for y in range(size):
        row = []
        for x in range(size):
            if (x - c) ** 2 + (y - c) ** 2 < (0.35 * size) ** 2:
                row.append((220, 180, 140))
            else:
                row.append((int(80 + 120 * x / size), int(60 + 140 * y / size), 200))
        img.append(row)
    return img


def load_face():
    root = tmp = None
    try:
        import tkinter as tk

        if os.path.exists(LOCAL):
            tmp = LOCAL
        else:
            req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f:
                tmp = f.name
                with urllib.request.urlopen(req, timeout=15) as r:
                    f.write(r.read())
        root = tk.Tk()
        root.withdraw()
        pic = tk.PhotoImage(file=tmp)
        w, h = pic.width(), pic.height()
        img = []
        for y in range(SIZE):
            sy = min(h - 1, y * h // SIZE)
            row = []
            for x in range(SIZE):
                sx = min(w - 1, x * w // SIZE)
                row.append(rgb(pic.get(sx, sy)))
            img.append(row)
        src = "local" if tmp == LOCAL else "download"
        return img, src
    except Exception as e:
        print("无法加载图片，使用内置测试图:", e)
        return fallback_color(SIZE), "fallback"
    finally:
        if root is not None:
            root.destroy()
        if tmp and tmp != LOCAL and os.path.isfile(tmp):
            os.remove(tmp)


def to_gray(rgb_img):
    return [[clamp(round(0.299 * r + 0.587 * g + 0.114 * b)) for r, g, b in row] for row in rgb_img]


def bmp(path, img):
    h, w = len(img), len(img[0])
    pad = (-3 * w) % 4
    raw_row = 3 * w + pad
    size = 54 + h * raw_row
    with open(path, "wb") as f:
        f.write(struct.pack("<2sIHHI", b"BM", size, 0, 0, 54))
        f.write(struct.pack("<IIIHHIIIIII", 40, w, h, 1, 24, 0, h * raw_row, 2835, 2835, 0, 0))
        pad_bytes = b"\x00" * pad
        for row in reversed(img):
            for r, g, b in row:
                f.write(bytes((b, g, r)))
            f.write(pad_bytes)


def gray_img(g):
    return [[(v, v, v) for v in row] for row in g]


def bilinear(g, y, x):
    h, w = len(g), len(g[0])
    y0, x0 = int(y), int(x)
    y1, x1 = min(y0 + 1, h - 1), min(x0 + 1, w - 1)
    dy, dx = y - y0, x - x0
    a = g[y0][x0] * (1 - dx) + g[y0][x1] * dx
    b = g[y1][x0] * (1 - dx) + g[y1][x1] * dx
    return a * (1 - dy) + b * dy


def rot_min(code):
    best = code
    for s in range(1, P):
        r = ((code >> s) | ((code << (P - s)) & 255)) & 255
        if r < best:
            best = r
    return best


def transitions(code):
    return sum(((code >> i) & 1) != ((code >> ((i + 1) % P)) & 1) for i in range(P))


# P=8 时：RI 共 36 类；U2 共 59 维（58 个 uniform + 1 个非 uniform）
ri_rep = sorted({rot_min(i) for i in range(256)})
ri_of = {v: i for i, v in enumerate(ri_rep)}
RI = [ri_of[rot_min(i)] for i in range(256)]
u2_list = [i for i in range(256) if transitions(i) <= 2]
U2 = [58] * 256
for k, v in enumerate(u2_list):
    U2[v] = k


def lbp(gray):
    h, w = len(gray), len(gray[0])
    raw = [[0] * w for _ in range(h)]
    ri = [[0] * w for _ in range(h)]
    u2 = [[0] * w for _ in range(h)]
    h0, h1, h2 = [0] * 256, [0] * 36, [0] * 59
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            c = gray[y][x]
            code = 0
            for i, (dy, dx) in enumerate(OFFSETS):
                if bilinear(gray, y + dy, x + dx) >= c:
                    code |= 1 << i
            raw[y][x] = code
            ri[y][x] = RI[code]
            u2[y][x] = U2[code]
            h0[code] += 1
            h1[RI[code]] += 1
            h2[U2[code]] += 1
    return raw, ri, u2, h0, h1, h2


def scale(mat, vmax):
    return [[round(v * 255 / vmax) if vmax else 0 for v in row] for row in mat]


def hist_img(hist):
    W, H = 800, 400
    pic = [[(255, 255, 255)] * W for _ in range(H)]
    L, Rm, T, B = 60, 20, 20, 40
    x0, x1, y0, y1 = L, W - Rm - 1, T, H - B - 1
    for y in range(y0, y1 + 1):
        pic[y][x0] = (0, 0, 0)
    for x in range(x0, x1 + 1):
        pic[y1][x] = (0, 0, 0)
    m = max(hist) or 1
    n = len(hist)
    for i, v in enumerate(hist):
        a = x0 + i * (x1 - x0 + 1) // n
        b = x0 + (i + 1) * (x1 - x0 + 1) // n - 1
        if b < a:
            b = a
        bar = 0 if v == 0 else max(1, round(v * (y1 - y0 - 1) / m))
        for y in range(y1 - bar, y1):
            for x in range(a, b + 1):
                pic[y][x] = (65, 105, 225)
    return pic


def save_csv(path, hist):
    with open(path, "w", encoding="utf-8") as f:
        f.write("bin_index,count\n")
        for i, v in enumerate(hist):
            f.write(f"{i},{v}\n")


def main():
    os.makedirs(OUT, exist_ok=True)
    color, src = load_face()
    g = to_gray(color)
    raw, ri, u2, h0, h1, h2 = lbp(g)

    n = (SIZE - 2) ** 2
    assert sum(h0) == sum(h1) == sum(h2) == n
    assert len(ri_rep) == 36 and len(u2_list) + 1 == 59

    bmp(os.path.join(OUT, "source_color.bmp"), color)
    bmp(os.path.join(OUT, "source_gray.bmp"), gray_img(g))
    bmp(os.path.join(OUT, "lbp_circular.bmp"), gray_img(raw))
    bmp(os.path.join(OUT, "lbp_ri.bmp"), gray_img(scale(ri, 35)))
    bmp(os.path.join(OUT, "lbp_u2.bmp"), gray_img(scale(u2, 58)))
    bmp(os.path.join(OUT, "hist_circular.bmp"), hist_img(h0))
    bmp(os.path.join(OUT, "hist_ri.bmp"), hist_img(h1))
    bmp(os.path.join(OUT, "hist_u2.bmp"), hist_img(h2))
    save_csv(os.path.join(OUT, "hist_circular.csv"), h0)
    save_csv(os.path.join(OUT, "hist_ri.csv"), h1)
    save_csv(os.path.join(OUT, "hist_u2.csv"), h2)

    print("图片来源:", src)
    print("有效像素:", n, "RI 箱数:", len(h1), "U2 箱数:", len(h2))
    print("输出目录:", os.path.abspath(OUT))


if __name__ == "__main__":
    main()
