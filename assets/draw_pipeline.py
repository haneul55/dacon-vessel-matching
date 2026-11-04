import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

OUT = sys.argv[1]
HAND, KO = "Segoe Print", "Malgun Gothic"
BLUE, RED, INK, GRAY = "#1f4e9c", "#a3201c", "#111111", "#7a7a7a"

with plt.xkcd(scale=0.8, length=120, randomness=1.5):
    plt.rcParams["path.effects"] = []
    fig, ax = plt.subplots(figsize=(13, 12), dpi=110)
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

    def box(x, y, w, h, fc="white", lw=2.2, ec=INK):
        ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=ec, lw=lw))

    def txt(x, y, s, c=INK, size=17, font=HAND, **kw):
        ax.text(x, y, s, color=c, fontsize=size, fontfamily=font, ha="center", va="center", **kw)

    def arrow(p, q, c=INK, rad=0.0, lw=2.2):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="->,head_width=5,head_length=9", color=c, lw=lw,
                                     connectionstyle=f"arc3,rad={rad}", mutation_scale=2))

    # input record
    box(3, 76, 27, 19, fc="#ece8e3", lw=0)
    box(7, 79, 19, 13)
    txt(16.5, 89.3, "금성8호", size=19, font=KO, weight="bold")
    txt(16.5, 85.2, "CS_9F3A...", size=12, c=GRAY)
    txt(16.5, 81.8, "1,234 GT · 파나마", size=11, c=GRAY, font=KO)
    txt(16.5, 72, "Port record", size=22)

    # retrieval arrow
    txt(41, 89, "Retrieve", size=20)
    arrow((33, 85.5), (50, 85.5), lw=2.8)

    # catalog: stacked cards, candidates highlighted
    for i, d in enumerate([2.4, 1.2, 0]):
        box(54 + d, 77 + d, 30, 17)
    for k in range(5):
        y = 90.5 - k * 2.6
        hl = k in (1, 3)
        box(56.5, y - 0.9, 25, 1.8, fc="#dde7f6" if hl else "white", lw=1.2, ec=BLUE if hl else GRAY)
    txt(69, 72, "Catalog  (78,595 vessels)", size=20)
    txt(69, 68.3, "name + tonnage ±5.5%  →  ~74 candidates", size=13, c=BLUE)
    txt(69, 65.3, "+ up to 50 nearest by LaBSE (±10% GT)", size=13, c=BLUE)

    # one candidate -> feature bar
    arrow((55, 76.3), (7.0, 49.5), c=GRAY, rad=0.0, lw=2.0)
    txt(23, 64, "one candidate", size=14, c=GRAY)
    segs = [("spelling", 10), ("sound", 9), ("hanja", 9), ("tonnage", 10), ("flag / type", 12), ("LaBSE", 10)]
    x0, yb, hb = 8, 47.5, 4.0
    xs = []
    x = x0
    for name, w in segs:
        box(x, yb, w, hb, lw=2.0)
        xs.append((x + w / 2, name))
        x += w
    bar_end = x
    for cx, name in xs:
        ax.plot([cx], [yb + hb / 2], "o", color=INK, ms=6)
        txt(cx, yb + hb + 3.4, name, size=13, c=BLUE)
    txt((x0 + bar_end) / 2, yb - 3.2, "68 features per candidate", size=15)

    # ranker
    arrow((bar_end + 1, yb + hb / 2), (bar_end + 7, yb + hb / 2), lw=2.4)
    box(bar_end + 8, yb - 3, 21, 10, fc="#dde7f6", ec=BLUE)
    txt(bar_end + 18.5, yb + 3.6, "LightGBM", size=15, c=BLUE)
    txt(bar_end + 18.5, yb + 0, "LambdaRank", size=15, c=BLUE)
    arrow((bar_end + 18.5, yb - 4), (bar_end + 18.5, yb - 10), c=RED)
    txt(bar_end + 18.5, yb - 13, "Ranking", size=17, c=RED)
    txt(bar_end + 18.5, yb - 16.5, "Loss", size=17, c=RED)
    txt(bar_end + 18.5, yb + 9.5, "Hangul / English rankers", size=12, c=GRAY)

    # bottom chain: top-5 -> cross-encoder -> call sign -> answer
    yc = 18
    arrow((bar_end + 8, yb - 2.5), (23, yc + 6.3), c=GRAY, rad=0.0, lw=2.0)
    txt(29, 35.5, "score every candidate, keep the best", size=13, c=GRAY)

    box(4, yc - 4, 20, 10)
    for k in range(5):
        box(6 + k * 3.4, yc - 1.5, 2.6, 5, fc="#dde7f6" if k == 0 else "white", lw=1.4, ec=BLUE if k == 0 else INK)
    txt(14, yc + 8.6, "Top-5", size=17, c=BLUE)

    arrow((25, yc + 1), (32, yc + 1), lw=2.4)
    box(33, yc - 4, 21, 10, fc="#dde7f6", ec=BLUE)
    txt(43.5, yc + 2.6, "Cross-encoder", size=15, c=BLUE)
    txt(43.5, yc - 1.0, "re-score", size=14, c=BLUE)
    txt(43.5, yc + 8.6, "Hangul rows only", size=13, c=GRAY)
    arrow((43.5, yc - 5), (43.5, yc - 9.5), c=RED)
    txt(43.5, yc - 12.3, "Listwise Loss", size=16, c=RED)

    arrow((55, yc + 1), (62, yc + 1), lw=2.4)
    box(63, yc - 4, 17, 10)
    txt(71.5, yc + 2.6, "Call-sign", size=15)
    txt(71.5, yc - 1.0, "rules", size=15)
    txt(71.5, yc + 8.6, "exclude · lookup · pool", size=12, c=GRAY)

    arrow((81, yc + 1), (86, yc + 1), lw=2.4)
    txt(93, yc + 3.0, "Jin Cheng 8", size=17, c=BLUE, weight="bold")
    txt(93, yc - 1.2, "(illustrative example)", size=11, c=GRAY)
    txt(93, yc - 6.5, "LB 0.95240", size=15, c=RED)

    # LaBSE training note
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.25, facecolor="white")
print("saved", OUT)
