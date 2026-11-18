import os
import numpy as np
import pandas as pd
from glob import glob
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from utils.viser_test import PoseViser

RAW_KEYPOINTS_DIR = os.path.join(ROOT, "data", "raw_keypoints")
OUT_DIR = os.path.join(ROOT, "data", "processed")

os.makedirs(OUT_DIR, exist_ok=True)


def load_raw_keypoints(npz_path):
    """Load BlazePose raw data (T,33,3)."""
    data = np.load(npz_path, allow_pickle=True)["data"]
    df = pd.DataFrame(data, columns=["frame","landmark","x","y","z","visibility"])

    pts_seq = []

    for f, group in df.groupby("frame"):
        group = group.sort_values("landmark")
        pts = group[["x","y","z"]].values  # (33,3)
        pts_seq.append(pts)

    return np.array(pts_seq)   # (T,33,3)


def create_height_corrected_target(raw_seq):    # 캐릭터 높이 올려줌. 좌표상 몸통이 원점인데 발끝을 바닥으로
    """
    raw_seq: (T,33,3)
    Return target height corrected (T,33,3)
    """

    seq = raw_seq.copy()

    # BlazePose foot joints
    LEFT_FOOT = 31
    RIGHT_FOOT = 32

    # 모든 프레임에서 발의 y값 중 최소값 → ground level
    foot_y = np.minimum(seq[:, LEFT_FOOT, 1], seq[:, RIGHT_FOOT, 1])
    
    # reshape to (T,1) → broadcast to (T,33)
    frame_ground = foot_y[:, None]

    # 전체 시퀀스를 ground=0으로 맞춤
    seq[:, :, 1] -= frame_ground

    # 땅 아래(-)로 떨어지지 않게 clamp
    seq[:, :, 1] = np.maximum(seq[:, :, 1], 0)

    return seq


# =========================================================
# [Assignment 3] 추가 후처리
# =========================================================

def moving_average(values, window=5):
    """1D 배열 이동평균. 양 끝은 끝값으로 채워서 길이 유지."""
    pad = window // 2
    padded = np.pad(values, pad, mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(padded, kernel, mode="valid")


def remove_spikes(raw_seq, window=5, threshold=0.1):
    """
    ① 튀는 프레임 제거
    주변 프레임(window개)의 중앙값과 threshold(m) 이상 차이나면 튄 값으로 보고,
    앞뒤 정상 프레임 값으로 선형 보간해서 채운다.
    """
    seq = raw_seq.copy()
    T, J, _ = seq.shape
    half = window // 2
    n_fixed = 0

    for j in range(J):
        # 주변 프레임 중앙값 계산
        median = np.array([
            np.median(seq[max(0, t - half):t + half + 1, j], axis=0)
            for t in range(T)
        ])                                                   # (T,3)
        diff = np.linalg.norm(seq[:, j] - median, axis=1)   # (T,)
        bad = diff > threshold
        good = ~bad
        n_fixed += bad.sum()

        if bad.any() and good.sum() >= 2:
            frames = np.arange(T)
            for k in range(3):   # x, y, z
                seq[bad, j, k] = np.interp(frames[bad], frames[good], seq[good, j, k])

    print(f"  [remove_spikes] fixed points: {n_fixed} / {T * J}")
    return seq


def smooth_motion(raw_seq, window=5):
    """② 모든 관절의 x, y, z에 이동평균 smoothing"""
    seq = raw_seq.copy()
    T, J, _ = seq.shape
    for j in range(J):
        for k in range(3):
            seq[:, j, k] = moving_average(seq[:, j, k], window)
    return seq


# (부모 관절, 자식 관절) - 몸통에서 바깥쪽 순서로 나열해야 함
BONES = [
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),   # left arm + hand
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),   # right arm + hand
    (23, 25), (25, 27), (27, 29), (27, 31),             # left leg + foot
    (24, 26), (26, 28), (28, 30), (28, 32),             # right leg + foot
]


def fix_bone_length(raw_seq):
    """
    ③ 뼈 길이 고정
    뼈의 방향은 그대로 두고, 길이만 전체 프레임 평균 길이로 맞춘다.
    부모가 먼저 고쳐지므로 자식은 '고쳐진 부모 위치 + 방향 * 평균 길이'로 다시 계산.
    """
    seq = raw_seq.copy()
    for parent, child in BONES:
        bone = raw_seq[:, child] - raw_seq[:, parent]               # 원래 뼈 벡터 (T,3)
        length = np.linalg.norm(bone, axis=1, keepdims=True)        # (T,1)
        mean_length = length.mean()
        direction = bone / np.maximum(length, 1e-8)                 # 단위 벡터
        seq[:, child] = seq[:, parent] + direction * mean_length
    return seq


def create_height_corrected_target_smooth(raw_seq, window=15):
    """
    기존 create_height_corrected_target과 같지만,
    프레임마다 구한 바닥 높이(foot_y)를 이동평균해서 사용한다.
    → 짧은 점프 순간에 몸이 바닥에 강제로 붙지 않고, 발 떨림도 줄어든다.
    """
    seq = raw_seq.copy()

    LEFT_FOOT = 31
    RIGHT_FOOT = 32

    foot_y = np.minimum(seq[:, LEFT_FOOT, 1], seq[:, RIGHT_FOOT, 1])
    foot_y = moving_average(foot_y, window)      # ← 기존 코드와 다른 부분

    seq[:, :, 1] -= foot_y[:, None]
    seq[:, :, 1] = np.maximum(seq[:, :, 1], 0)
    return seq


def post_process(raw_seq):
    """② → ③ → 높이 보정(smooth) 순서로 실행"""
    # ① 튀는 값 보정은 사용하지 않음: 서브 임팩트 순간의 빠른 움직임(속도감)을 살리기 위해
    # seq = remove_spikes(raw_seq, window=5, threshold=0.1)
    seq = smooth_motion(raw_seq, window=5)   
    seq = fix_bone_length(seq)
    seq = create_height_corrected_target_smooth(seq, window=15)
    return seq


def export_unity_csv(motion, output_path):
    """
    motion: (T, 33, 3)
    Export Unity-compatible CSV.
    """

    T, n_landmarks, _ = motion.shape

    rows = []

    for t in range(T):
        for landmark in range(n_landmarks):

            x, y, z = motion[t, landmark]

            rows.append([
                t,
                landmark,
                float(x),
                float(y),
                float(z),
                1.0
            ])

    df = pd.DataFrame(
        rows,
        columns=[
            "frame",
            "landmark",
            "x",
            "y",
            "z",
            "visibility"
        ]
    )

    # 문자열 경로 처리
    output_path = os.path.abspath(output_path)

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    df.to_csv(
        output_path,
        index=False,
        encoding="utf-8"
    )

    print(f"Unity CSV saved: {output_path}")
    print(f"Frames: {T}")
    print(f"Landmarks: {n_landmarks}")
    print(f"Total rows: {len(df)}")

def main():
    paths = glob(f"{RAW_KEYPOINTS_DIR}/*.npz")
    print(f"Found {len(paths)} raw keypoint files")

    for path in paths:
        base = os.path.basename(path).replace(".npz", "")

        # -------------------------------
        # 1) Load raw BlazePose sequence
        # -------------------------------
        raw_seq = load_raw_keypoints(path)      # (T,33,3)
        raw_out_path = os.path.join(OUT_DIR, f"{base}_raw.npy")
        np.save(raw_out_path, raw_seq)
        print(f"Saved raw → {raw_out_path}")

        # ---------------------------------------
        # 2) Create height-corrected target (GT)
        # ---------------------------------------
        target_seq = create_height_corrected_target(raw_seq)
        target_out_path = os.path.join(OUT_DIR, f"{base}_target.npy")
        np.save(target_out_path, target_seq)
        print(f"Saved target → {target_out_path}")

        OUTPUT_PATH = (
                os.path.join(OUT_DIR,
                f"{base}_target.csv")
            )
        export_unity_csv(target_seq,OUTPUT_PATH)

        # ---------------------------------------
        # 3) [Assignment 3] 추가 후처리 결과 (post)
        # ---------------------------------------
        post_seq = post_process(raw_seq.astype(float))
        post_out_path = os.path.join(OUT_DIR, f"{base}_post.npy")
        np.save(post_out_path, post_seq)
        print(f"Saved post → {post_out_path}")
        export_unity_csv(post_seq, os.path.join(OUT_DIR, f"{base}_post.csv"))

    print("\nDone processing all keypoint files.")

    # IF you want to visualize the output
    vis = PoseViser(fps=30)
    # raw = np.load("./data/processed/2018_Roger_Federer_Indian_Wells_Practice_The_Perfect_Serve_Technique_target.npy")  # 기존 보정 결과
    raw = np.load("./data/processed/2018_Roger_Federer_Indian_Wells_Practice_The_Perfect_Serve_Technique_post.npy")  # 파일명 똑같이 변경 (추가 후처리 결과)
    print(raw.shape)
    vis.play_sequence(raw)


if __name__ == "__main__":
    main()
