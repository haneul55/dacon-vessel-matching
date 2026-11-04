"""Model architecture figure for the README: stage overview (top) and what each stage does to one record (bottom).
usage: python draw_architecture.py <out.png>"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, Polygon, Rectangle

OUT = sys.argv[1]
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
BLUE, LINE, FILL, TXT, MUTED = "#2f75b5", "#4d4d4d", "#e4e2dd", "#3a3a3a", "#7a7a7a"

fig, ax = plt.subplots(figsize=(18, 10.4), dpi=110)
ax.set_xlim(0, 180); ax.set_ylim(0, 104); ax.axis("off")


def rect(x, y, w, h, fc="white", ec=LINE, lw=1.0, z=1):
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=ec, lw=lw, zorder=z))


def text(x, y, s, size=10.5, c=TXT, ha="center", **kw):
    ax.text(x, y, s, fontsize=size, color=c, ha=ha, va="center", zorder=5, linespacing=1.25, **kw)


def arrow(p, q, c=BLUE, lw=2.0, ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=12, color=c, lw=lw, ls=ls,
                                 shrinkA=0, shrinkB=0, zorder=4))


def elbow(pts, c=BLUE, lw=2.0):
    xs, ys = zip(*pts)
    ax.plot(xs[:-1], ys[:-1], color=c, lw=lw, zorder=4, solid_capstyle="butt")
    arrow(pts[-2], pts[-1], c=c, lw=lw)


# ---------------- input record ----------------
text(12, 101, "입력 기록 1행", size=11)
rect(3, 84, 18, 14, fc="#d9d6d0", ec="none")
rect(5, 86, 14, 10)
text(12, 93.5, "금성8호", size=13, weight="bold")
text(12, 90.2, "CS_9F3A…", size=8.5, c=MUTED)
text(12, 87.7, "1,234 GT · 파나마", size=8.5, c=MUTED)

# ---------------- top row: stage overview ----------------
W, GAP, X0, YT, HT = 21.5, 5, 2, 50, 26
stages = [
    ("A  정규화", ["한글 정리", "발음 키 · 자모 키", "한자음 · 뜻 키"]),
    ("B  후보 검색", ["문자열 유사도\nTop-40", "톤수 ±5.5%\nTop-30", "한자음 사전 조회"]),
    ("C  의미 검색", ["LaBSE 인코더", "한글↔영문\n파인튜닝", "톤수 ±10%\nTop-50"]),
    ("D  특징 계산", ["철자 유사도", "발음 · 한자음", "톤수 · 국적 · 임베딩"]),
    ("E  순위 결정", ["LightGBM\nLambdaRank", "한글 3개 / 영문 9개\n평균", "Cross-encoder\n한글 상위 5개"]),
    ("F  후처리", ["호출부호 배제", "호출부호 고정 조회", "같은 호출부호 묶기"]),
]
xs = [X0 + i * (W + GAP) for i in range(len(stages))]
for x, (title, rows) in zip(xs, stages):
    rect(x, YT, W, HT, lw=1.1)
    text(x + W / 2, YT + HT + 2.4, title, size=11.5, weight="bold")
    hb = (HT - 4 * 1.4) / 3
    for k, r in enumerate(rows):
        y = YT + HT - 1.4 - (k + 1) * hb - k * 1.4
        rect(x + 1.4, y, W - 2.8, hb, fc=FILL, ec="none")
        text(x + W / 2, y + hb / 2, r, size=9.5)
for a, b in zip(xs[:-1], xs[1:]):
    arrow((a + W, YT + HT / 2), (b, YT + HT / 2))
arrow((20, 84), (20, YT + HT))

# output of top row
ox = xs[-1] + W
arrow((ox, YT + HT / 2), (ox + 5, YT + HT / 2))
text(ox + 6, YT + HT / 2 + 1.6, "VESSEL_26017", size=10.5, ha="left", weight="bold")
text(ox + 6, YT + HT / 2 - 1.6, "Jin Cheng 8", size=10.5, ha="left", c=BLUE)

# ---------------- bottom row: one record through each stage ----------------
YB, HB = 14, 24
for x in xs:
    rect(x, YB, W, HB, lw=1.1)

# A detail: three keys
x = xs[0]
text(x + W / 2, YB + HB - 3, "금성8호", size=11, weight="bold")
for k, (s, lab) in enumerate([("geum seong 8", "발음"), ("ㄱㅡㅁㅅㅓㅇ 8", "자모"), ("Jin Cheng 8", "한자음")]):
    y = YB + HB - 8 - k * 4.6
    rect(x + 1.5, y - 1.7, 5.5, 3.4, fc=FILL, ec="none")
    text(x + 4.25, y, lab, size=8.5)
    text(x + 8, y, s, size=9.5, ha="left", c=BLUE)

# B detail: catalog grid with a few retrieved
x = xs[1]
rng = np.random.default_rng(3)
hit = set(rng.choice(48, 7, replace=False))
for i in range(48):
    r, c = divmod(i, 8)
    rect(x + 2 + c * 2.25, YB + HB - 5 - r * 2.25, 1.8, 1.8, fc=BLUE if i in hit else FILL, ec="none")
text(x + W / 2, YB + 3.6, "78,595척 → 평균 74척", size=9.5)

# C detail: embedding space
x = xs[2]
cx, cy = x + W / 2, YB + HB / 2 + 2
pts = rng.normal(0, 1, (40, 2)) * [4.2, 3.3]
pts = pts[(np.abs(pts[:, 0]) < 8.5) & (pts[:, 1] > -5.5) & (pts[:, 1] < 7)][:28]
for (px, py) in pts:
    near = px * px / 16 + py * py / 10 < 0.5
    ax.plot(cx + px, cy + py, "o", ms=4.2, color=BLUE if near else "#b9b6b0", zorder=3)
ax.add_patch(Circle((cx, cy), 3.0, fill=False, ec=BLUE, lw=1.0, ls="--", zorder=3))
ax.plot(cx, cy, "*", ms=13, color="#d9822b", zorder=4)
text(x + W / 2, YB + 3.6, "뜻이 가까운 배 추가", size=9.5)

# D detail: N x 68 matrix
x = xs[3]
for r in range(6):
    for c in range(7):
        v = rng.random()
        shade = BLUE if (r == 1 and c in (0, 1, 2, 5)) else ("#c9c6bf" if v > 0.5 else FILL)
        rect(x + 2.2 + c * 2.15, YB + HB - 5 - r * 2.25, 1.8, 1.8, fc=shade, ec="none")
text(x + W - 0.6, YB + HB - 5 - 1 * 2.25 + 0.9, "◀정답", size=8, c=BLUE, ha="right")
text(x + W / 2, YB + 3.6, "후보 N × 특징 68", size=9.5)

# E detail: trees + cross-encoder
x = xs[4]
for k in range(3):
    tx = x + 4 + k * 5
    ty = YB + HB - 4
    ax.add_patch(Polygon([[tx, ty], [tx - 2, ty - 4.2], [tx + 2, ty - 4.2]], closed=True, fc=FILL, ec=LINE, lw=0.8, zorder=3))
text(x + 17.5, YB + HB - 6.2, "…", size=12)
text(x + W / 2, YB + HB - 10.6, "점수 = 랭커", size=9.5)
text(x + W / 2, YB + HB - 13.8, "+ 0.25 × Cross-encoder", size=9.5, c=BLUE)
text(x + W / 2, YB + 3.6, "후보마다 점수", size=9.5)

# F detail: call sign check
x = xs[5]
text(x + W / 2, YB + HB - 3.5, "CS_9F3A…", size=10, weight="bold")
text(x + W / 2, YB + HB - 7.6, "train에 있음?", size=9.5)
text(x + W / 2, YB + HB - 11.2, "예 → 그 배로 확정", size=9, c=BLUE)
text(x + W / 2, YB + HB - 14.4, "아니오 → train 배 제외", size=9, c=BLUE)
text(x + W / 2, YB + 3.6, "규칙마다 자기 검증", size=9.5)

# probability bars (example)
px0 = ox + 6
text(px0 + 6, YB + HB + 1.2, "최종 확률 (예시)", size=10)
bars = [("Jin Cheng 8", 0.91), ("Jin Sheng 8", 0.05), ("Kum Sung 8", 0.03), ("Jin Cheng 18", 0.01)]
for k, (n, p) in enumerate(bars):
    y = YB + HB - 3.5 - k * 5.2
    text(px0 + 0.4, y + 1.4, n, size=9, ha="left")
    rect(px0 + 0.4, y - 1.6, max(p * 20, 0.4), 1.6, fc=BLUE, ec="none")
ax.plot([px0 - 0.4, px0 - 0.4], [YB - 1, YB + HB - 1], color=LINE, lw=0.8)
arrow((ox, YB + HB / 2), (px0 - 1, YB + HB / 2))

# intermediate outputs under the bottom row (like feature maps)
outs = ["키 3종", "후보 ~74척", "+ 의미 후보", "N×68 행렬", "후보별 점수", "후보 정리"]
for i, (x, lab) in enumerate(zip(xs, outs)):
    sx = x + W + GAP / 2 - 1.6 if i < 5 else None
    if sx is None:
        continue
    for d in (1.2, 0.6, 0):
        rect(sx - 1.6 + d, 4.5 + d, 3.2, 3.2, fc=FILL, ec=LINE, lw=0.7, z=3)
    elbow([(x + W * 0.72, YB), (x + W * 0.72, 6.1), (sx - 1.6, 6.1)], lw=1.6)
    elbow([(sx + 2.8, 6.1), (xs[i + 1] + W * 0.28, 6.1), (xs[i + 1] + W * 0.28, YB)], lw=1.6)
    text(sx + 0.3, 1.6, lab, size=8.5, c=MUTED)

text(2, 46, "위: 단계 구성   아래: 기록 한 줄이 각 단계에서 바뀌는 모습", size=9.5, c=MUTED, ha="left")
fig.savefig(OUT, bbox_inches="tight", pad_inches=0.3, facecolor="white")
print("saved", OUT)
