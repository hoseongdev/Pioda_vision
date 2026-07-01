"""
feature_extractor.py — 영상 → Py-Feat → 프레임별 표정 지표(Action Unit) 시계열

[방식]
    Py-Feat(MIT 라이선스)의 Detectorv2로 영상에서 프레임별 Action Unit(AU)을 추출한다.
    AU는 FACS 기반 표정 근육 단위 활성도로, 파킨슨/치매의 안면 동결(hypomimia) 연구에서
    가장 널리 쓰이는 임상 지표. 치매 징후는 이 AU들의 '시간당 변동 폭'이 줄어드는 특징.

[출력]  (T, 20) — 프레임별 AU 20종 활성도 (0~1)
    AU01 내측눈썹올림 / AU02 외측눈썹올림 / AU04 눈썹내림 / AU06 볼올림(진짜웃음)
    AU12 입꼬리당김(미소) / AU15 입꼬리내림 / AU25 입벌림 ... 등

[참고]  온디바이스 제약을 두지 않기로 함(멘토링 반영) → 무겁지만 정확한 Py-Feat 사용.
"""

import glob
import numpy as np

# Py-Feat AU 컬럼 (Detectorv2 출력 순서 고정)
AU_COLUMNS = ["AU01", "AU02", "AU04", "AU05", "AU06", "AU07", "AU09", "AU10",
              "AU11", "AU12", "AU14", "AU15", "AU17", "AU20", "AU23", "AU24",
              "AU25", "AU26", "AU28", "AU43"]
FEATURE_NAMES = AU_COLUMNS
NUM_FEATURES = len(AU_COLUMNS)

# Detector는 생성 비용이 크므로 모듈 전역에 한 번만 로드 (lazy)
_detector = None


def _get_detector(device="cpu"):
    global _detector
    if _detector is None:
        from feat import Detectorv2
        print(f"[extractor] Py-Feat Detectorv2 로드 중 (device={device})...")
        _detector = Detectorv2(device=device)
        print("[extractor] Detector 준비 완료")
    return _detector


def extract_au_series(video_path, skip_frames=10, device="cpu"):
    """영상 → 프레임별 AU 시계열 (T, 20). skip_frames: N프레임마다 1장 분석."""
    detector = _get_detector(device)
    fex = detector.detect(video_path, data_type="video", skip_frames=skip_frames)

    # AU 컬럼만 추출, 얼굴 미검출(NaN) 프레임 제거
    au = fex[AU_COLUMNS].to_numpy(dtype=np.float32)
    valid = ~np.isnan(au).any(axis=1)
    au = au[valid]

    if len(au) == 0:
        print(f"[extractor] 유효 프레임 없음: {video_path}")
        return None

    print(f"[extractor] {video_path}: {len(au)}프레임 AU 추출 완료")
    return au


def build_sequences(series, seq_length=20, stride=10):
    """AU 시계열 (T, 20) → 고정 길이 윈도우 (N, seq_length, 20)"""
    if series is None or len(series) < seq_length:
        return None
    windows = [series[i:i + seq_length]
               for i in range(0, len(series) - seq_length + 1, stride)]
    return np.array(windows, dtype=np.float32) if windows else None


def build_dataset_from_dir(data_dir="data", seq_length=20, stride=10,
                           skip_frames=10, device="cpu"):
    """data/ 안 모든 영상 → AU 추출 → 윈도우 시퀀스 합치기 (N, seq_length, 20)"""
    paths = sorted(glob.glob(f"{data_dir}/**/*.mp4", recursive=True))
    print(f"[extractor] 영상 {len(paths)}개 발견")

    all_windows = []
    for p in paths:
        series = extract_au_series(p, skip_frames=skip_frames, device=device)
        seqs = build_sequences(series, seq_length=seq_length, stride=stride)
        if seqs is not None:
            all_windows.append(seqs)

    if not all_windows:
        print("[extractor] 시퀀스 생성 실패: 유효 영상 부족")
        return None

    dataset = np.concatenate(all_windows, axis=0)
    print(f"[extractor] 최종 데이터셋: {dataset.shape}  (N, seq_len={seq_length}, AU={NUM_FEATURES})")
    return dataset


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/test.mp4"
    arr = extract_au_series(path)
    if arr is not None:
        print("\nAU별 [평균 / 표준편차(=변동폭)]:")
        for i, name in enumerate(AU_COLUMNS):
            print(f"  {name}: {arr[:, i].mean():.4f} / {arr[:, i].std():.4f}")
        print("\n※ 표준편차가 작을수록 '표정이 굳어있다(안면 동결)'는 신호")
