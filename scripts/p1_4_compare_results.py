# scripts/p1_4_compare_results.py
# Assignment 3 - 4. Results & Analysis
# before(제공된 후처리, _target) vs after(추가 후처리, _post) 비교 그래프

import os
import numpy as np
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(ROOT, "data", "processed")
FIG_DIR = os.path.join(ROOT, "outputs", "assignment3")
os.makedirs(FIG_DIR, exist_ok=True)

NAME = "2018_Roger_Federer_Indian_Wells_Practice_The_Perfect_Serve_Technique"  # 파일명 똑같이 변경

# BlazePose 관절 번호
R_ELBOW, R_WRIST = 14, 16
L_FOOT, R_FOOT = 31, 32

BEFORE_COLOR = "tab:gray"
AFTER_COLOR = "tab:red"


def bone_length(seq, a, b):
    """두 관절 사이 거리 (T,)"""
    return np.linalg.norm(seq[:, a] - seq[:, b], axis=1)


def joint_speed(seq):
    """프레임마다 관절이 움직인 거리 (T-1, 33)"""
    return np.linalg.norm(seq[1:] - seq[:-1], axis=2)


def main():
    before = np.load(os.path.join(PROCESSED_DIR, f"{NAME}_target.npy")).astype(float)  # (T,33,3)
    after = np.load(os.path.join(PROCESSED_DIR, f"{NAME}_post.npy")).astype(float)     # (T,33,3)
    frames = np.arange(len(before))

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # -------------------------------------------------
    # (1) 오른 손목 x 좌표: 떨림이 줄었는지
    # -------------------------------------------------
    ax = axes[0, 0]
    ax.plot(frames, before[:, R_WRIST, 0], color=BEFORE_COLOR, label="before")
    ax.plot(frames, after[:, R_WRIST, 0], color=AFTER_COLOR, label="after")
    ax.set_title("(1) Right wrist x position")
    ax.set_xlabel("frame")
    ax.set_ylabel("position (m)")
    ax.legend()

    # -------------------------------------------------
    # (2) 프레임 간 이동 거리 (전체 관절 평균): 떨림 정도
    # -------------------------------------------------
    ax = axes[0, 1]
    speed_before = joint_speed(before).mean(axis=1)
    speed_after = joint_speed(after).mean(axis=1)
    ax.plot(frames[1:], speed_before, color=BEFORE_COLOR,
            label=f"before (mean {speed_before.mean():.4f})")
    ax.plot(frames[1:], speed_after, color=AFTER_COLOR,
            label=f"after (mean {speed_after.mean():.4f})")
    ax.set_title("(2) Frame-to-frame movement (all joints mean)")
    ax.set_xlabel("frame")
    ax.set_ylabel("distance per frame (m)")
    ax.legend()

    # -------------------------------------------------
    # (3) 오른 아래팔 길이: 일정해졌는지
    # -------------------------------------------------
    ax = axes[1, 0]
    len_before = bone_length(before, R_ELBOW, R_WRIST)
    len_after = bone_length(after, R_ELBOW, R_WRIST)
    ax.plot(frames, len_before, color=BEFORE_COLOR,
            label=f"before (std {len_before.std():.3f})")
    ax.plot(frames, len_after, color=AFTER_COLOR,
            label=f"after (std {len_after.std():.3f})")
    ax.set_title("(3) Right forearm length")
    ax.set_xlabel("frame")
    ax.set_ylabel("length (m)")
    ax.legend()

    # -------------------------------------------------
    # (4) 더 낮은 발의 높이: 항상 0 → 떠 있는 순간 생김
    # -------------------------------------------------
    ax = axes[1, 1]
    foot_before = np.minimum(before[:, L_FOOT, 1], before[:, R_FOOT, 1])
    foot_after = np.minimum(after[:, L_FOOT, 1], after[:, R_FOOT, 1])
    ax.plot(frames, foot_before, color=BEFORE_COLOR, label="before (always 0)")
    ax.plot(frames, foot_after, color=AFTER_COLOR, label="after")
    ax.set_title("(4) Lower foot height (ground contact)")
    ax.set_xlabel("frame")
    ax.set_ylabel("height (m)")
    ax.legend()

    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "before_after_comparison.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved figure → {out_path}")

    # 숫자로도 정리 (PPT 표에 쓰기)
    wrist_before = joint_speed(before)[:, R_WRIST].max()
    wrist_after = joint_speed(after)[:, R_WRIST].max()
    print("\n                         before    after")
    print(f"Mean movement (m/frame)  {speed_before.mean():.4f}    {speed_after.mean():.4f}")
    print(f"Wrist max move (m/frame) {wrist_before:.3f}     {wrist_after:.3f}")
    print(f"Forearm length std (m)   {len_before.std():.4f}    {len_after.std():.4f}")
    print(f"Frames both feet in air  {(foot_before > 0.01).sum():<9} {(foot_after > 0.01).sum()}")

    plt.show()


if __name__ == "__main__":
    main()
