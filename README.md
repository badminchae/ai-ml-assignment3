# 3D Motion Data Post-Processing (Assignment 3)

유튜브 영상에서 MediaPipe BlazePose로 3D 관절 데이터를 추출하고,
문제를 확인한 뒤 후처리(post-processing)로 품질을 개선한 과제입니다.
결과 CSV는 Unity 캐릭터에 넣어 영상 속 동작을 따라 하게 할 수 있습니다.

- 사용 영상: *2018 Roger Federer Indian Wells Practice – The Perfect Serve Technique* (YouTube Shorts)
- 데이터: 320 frames · 30 fps · 관절 33개 (x, y, z)

## 폴더 구조

```
.
├─ scripts/
│  ├─ p1_create_raw_keypoints.py   # 1. 유튜브 다운로드 + 3D 관절 추출 (.npz)
│  ├─ p1_2_post_processing.py      # 3. 후처리 (기존 높이 보정 + 추가 후처리) → Unity CSV
│  ├─ p1_3_visualize_problems.py   # 2. 문제 확인 그래프
│  ├─ p1_4_compare_results.py      # 4. 후처리 전후 비교 그래프
│  ├─ p2_create_dataset.py         # (final project) 동작 분류 데이터셋 생성
│  ├─ p3_train_mlp.py              # (final project) MLP 학습
│  └─ p4_predict.py                # (final project) 동작 예측
├─ utils/viser_test.py             # viser 3D 시각화 (http://localhost:8080)
├─ data/
│  ├─ raw_videos/                  # 다운로드한 영상
│  ├─ raw_keypoints/               # 추출한 관절 데이터 (.npz)
│  └─ processed/                   # _raw.npy, _target(.npy/.csv), _post(.npy/.csv)
├─ outputs/
│  ├─ assignment3/                 # 문제 확인 / 전후 비교 그래프
│  └─ Assignment3_..._report.pptx  # 보고서
└─ environment.yml
```

## 실행 방법

conda 환경을 만들고 저장소 최상위 폴더에서 순서대로 실행합니다.

```bash
conda env create -f environment.yml
conda activate ai_ml_final

python scripts/p1_create_raw_keypoints.py   # 영상 다운로드 + 관절 추출
python scripts/p1_2_post_processing.py      # 후처리 + CSV 저장 + viser 재생
python scripts/p1_3_visualize_problems.py   # 문제 확인 그래프
python scripts/p1_4_compare_results.py      # 전후 비교 그래프
```

> 다른 영상을 쓰려면 `p1_create_raw_keypoints.py`의 `url_list`와
> `p1_2_post_processing.py`, `p1_3`, `p1_4`의 파일명(`NAME`)을 바꿔 주세요.

## 문제 → 해결 → 결과

| 문제 | 해결 방법 | 결과 |
|---|---|---|
| 관절 떨림·튐 (손목이 1프레임에 0.84 m 이동) | 이동평균 smoothing (창 5) | 평균 프레임 간 이동 0.0387 → 0.0228 m (−41%) |
| 뼈 길이 변화 (아래팔 0.06 ~ 0.30 m) | 뼈 방향은 유지, 길이만 평균값으로 고정 | 아래팔 길이 std 0.029 → 0.000 m |
| 점프 소실 (매 프레임 낮은 발을 y=0에 고정) | 바닥 높이도 이동평균 (창 15) | 두 발 뜬 프레임 0 → 68 (최대 약 6 cm) |

후처리는 `p1_2_post_processing.py`에서 **기존 함수는 그대로 두고 함수를 추가**했습니다.

- `smooth_motion` / `moving_average`: 관절 좌표 이동평균
- `fix_bone_length`: 뼈 길이 고정
- `create_height_corrected_target_smooth`: 바닥 높이 smoothing을 쓰는 높이 보정
- `post_process`: 위 순서대로 실행
- `remove_spikes`: 튀는 값 보간. 서브 임팩트의 속도감을 살리기 위해 **사용하지 않음**

## 한계

- smoothing 때문에 임팩트 순간 손목 속도가 줄어듦 (0.84 → 0.28 m/frame)
- BlazePose 3D 좌표는 엉덩이 중심이라 실제 점프 높이는 복원할 수 없음
- 바닥 smoothing의 부작용으로 발을 바꿔 디딜 때 1~3 cm 뜨기도 함

## Unity 시각화

Unity 프로젝트는 용량이 커서 이 저장소에 넣지 않았습니다.
Unity의 CSV 로더(`Assets/Scripts/MotionDataFile/BlazePoseDataFile.cs`)가 자체적으로
5프레임 이동평균을 걸고 있어서, Python 후처리 결과를 그대로 보기 위해
`ImportCSVData` 안의 smoothing 블록을 `/* ... */`로 주석 처리했습니다.

Unity에서는 Play → `World` 선택 → **CSV Loader: Load CSV File Directory**로
`data/processed` 폴더 선택 → 파일 번호 입력 → **reset & play animation** 순서로 재생합니다.
