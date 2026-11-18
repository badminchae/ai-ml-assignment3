# scripts/p1_3_visualize_problems.py
# Assignment 3 - 2. Problem Identification
# p1_2에서 저장한 raw / target 데이터를 그래프로 그려서 문제를 확인한다.

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


def bone_length(seq, a, b):
    """두 관절 사이 거리 (T,)"""
    return np.linalg.norm(seq[:, a] - seq[:, b], axis=1)


def joint_speed(seq):
    """프레임마다 관절이 움직인 거리 (T-1, 33)"""
    return np.linalg.norm(seq[1:] - seq[:-1], axis=2)


def main():
    raw = np.load(os.path.join(PROCESSED_DIR, f"{NAME}_raw.npy")).astype(float)       # (T,33,3)
    target = np.load(os.path.join(PROCESSED_DIR, f"{NAME}_target.npy")).astype(float) # (T,33,3)
    T = len(raw)
    frames = np.arange(T)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # -------------------------------------------------
    # (1) Jitter: 오른쪽 손목 좌표가 프레임마다 튀는지
    # -------------------------------------------------
    ax = axes[0, 0]
    for k, axis_name in enumerate(["x", "y", "z"]):
        ax.plot(frames, raw[:, R_WRIST, k], label=axis_name)
    ax.set_title("(1) Right wrist position (raw) - jitter")
    ax.set_xlabel("frame")
    ax.set_ylabel("position (m)")
    ax.legend()

    # -------------------------------------------------
    # (2) Jitter: 프레임 간 이동 거리 (튀는 프레임 찾기)
    # -------------------------------------------------
    ax = axes[0, 1]
    speed = joint_speed(raw)            # (T-1,33)
    mean_speed = speed.mean(axis=1)     # 프레임별 전체 관절 평균
    ax.plot(frames[1:], speed[:, R_WRIST], label="right wrist", alpha=0.8)
    ax.plot(frames[1:], mean_speed, label="all joints (mean)", color="black")
    ax.set_title("(2) Frame-to-frame movement - sudden spikes")
    ax.set_xlabel("frame")
    ax.set_ylabel("distance per frame (m)")
    ax.legend()

    # -------------------------------------------------
    # (3) 뼈 길이 변화: 원래는 일정해야 함
    # -------------------------------------------------
    ax = axes[1, 0]
    forearm = bone_length(raw, R_ELBOW, R_WRIST)
    ax.plot(frames, forearm, label="right forearm length")
    ax.axhline(forearm.mean(), color="red", linestyle="--",
               label=f"mean = {forearm.mean():.3f} m")
    ax.set_title("(3) Bone length is not constant")
    ax.set_xlabel("frame")
    ax.set_ylabel("length (m)")
    ax.legend()

    # -------------------------------------------------
    # (4) 높이 보정 후 발 높이: 한 발이 항상 0 → 점프가 사라짐
    # -------------------------------------------------
    ax = axes[1, 1]
    ax.plot(frames, target[:, L_FOOT, 1], label="left foot y")
    ax.plot(frames, target[:, R_FOOT, 1], label="right foot y")
    ax.plot(frames, np.minimum(target[:, L_FOOT, 1], target[:, R_FOOT, 1]),
            label="lower foot y", color="black", linestyle="--")
    ax.set_title("(4) After height correction - one foot always at y=0")
    ax.set_xlabel("frame")
    ax.set_ylabel("height (m)")
    ax.legend()

    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "problem_identification.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved figure → {out_path}")

    # 숫자로도 정리 (PPT에 쓰기)
    print(f"Frames: {T}")
    print(f"Right forearm length: mean {forearm.mean():.3f}, "
          f"min {forearm.min():.3f}, max {forearm.max():.3f}, std {forearm.std():.3f}")
    print(f"Max movement in one frame (right wrist): {speed[:, R_WRIST].max():.3f} m")
    print(f"Lower foot y after correction: always {np.minimum(target[:, L_FOOT, 1], target[:, R_FOOT, 1]).max():.3f}")

    plt.show()


if __name__ == "__main__":
    main()
