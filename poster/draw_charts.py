"""Charts for the DS-Day poster. usage: python draw_charts.py <out_dir>"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(sys.argv[1])
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
NAVY, BLUE, GRAY = "#1f3a68", "#2f75b5", "#b9b6b0"

# leaderboard: each step's score (decoded correct counts in the label)
steps = [("기본 파이프라인\n(랭커 + cross-encoder)", 0.95147, "한글 504\n영문 3,429"),
         ("+ 영문 랭커\n3개 → 9개 앙상블", 0.95181, "한글 504\n영문 3,433"),
         ("+ 호출부호\n배제 · 묶기", 0.95206, "한글 504\n영문 3,436"),
         ("최종\n(두 가지 모두)", 0.95240, "한글 504\n영문 3,440")]
fig, ax = plt.subplots(figsize=(8.6, 4.6), dpi=220)
xs = range(len(steps))
bars = ax.bar(xs, [s[1] for s in steps], color=[GRAY, BLUE, BLUE, NAVY], width=0.62)
for x, (_, v, lab) in zip(xs, steps):
    ax.text(x, v + 0.00004, f"{v:.5f}", ha="center", va="bottom", fontsize=12, fontweight="bold", color=NAVY)
    ax.text(x, 0.95095 + 0.00003, lab, ha="center", va="bottom", fontsize=9.5, color="white", linespacing=1.3)
ax.set_xticks(list(xs), [s[0] for s in steps], fontsize=10)
ax.set_ylim(0.9509, 0.9528)
ax.set_ylabel("Public LB", fontsize=11)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.grid(axis="y", alpha=0.3)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(OUT / "lb_steps.png", facecolor="white")

# stress test (development version K2): Hangul vs English accuracy per corruption
conds = [("원본", 99.66, 99.95), ("톤수 제거", 99.58, 99.75), ("자모 오타", 99.50, 99.95),
         ("이름만", 97.40, 99.75), ("복합 손상", 86.90, 96.20)]
fig, ax = plt.subplots(figsize=(8.6, 4.2), dpi=220)
w = 0.38
for i, (n, h, e) in enumerate(conds):
    ax.bar(i - w / 2, h, w, color=NAVY, label="한글 기록" if i == 0 else None)
    ax.bar(i + w / 2, e, w, color=GRAY, label="영문 기록" if i == 0 else None)
    ax.text(i - w / 2, h + 0.3, f"{h:.2f}", ha="center", fontsize=9, color=NAVY)
    ax.text(i + w / 2, e + 0.3, f"{e:.2f}", ha="center", fontsize=9, color="#555")
ax.set_xticks(range(len(conds)), [c[0] for c in conds], fontsize=11)
ax.set_ylim(80, 102)
ax.set_ylabel("정확도 (%)", fontsize=11)
ax.legend(frameon=False, fontsize=10, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.13))
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.grid(axis="y", alpha=0.3)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(OUT / "stress.png", facecolor="white")
print("saved")
