"""
inference.py — 영상에 대한 안면 이상 탐지 (재구성 오차 기반)

음성 모듈의 inference.py와 동일한 인터페이스.
앙상블 시 이 모듈의 score를 영상 이상 점수로 사용한다.

사용 예시:
    python inference.py data/test.mp4

또는:
    from inference import VisionAnomalyDetector
    detector = VisionAnomalyDetector()
    result = detector.score_video("data/test.mp4")
    # result = {"video_score": float, "is_anomaly": bool, "n_windows": int}
"""

import sys
import numpy as np
import torch

from model_train import VisionLSTMAutoencoder
from feature_extractor import extract_au_series, build_sequences, NUM_FEATURES


class VisionAnomalyDetector:
    def __init__(
        self,
        model_path="vision_autoencoder.pt",
        norm_path="norm_stats.pt",
        threshold_path="threshold.pt",
        seq_length=15,
        stride=5,
    ):
        norm = torch.load(norm_path, weights_only=True)
        self.mean = norm["mean"]
        self.std  = norm["std"]
        self.threshold = torch.load(threshold_path, weights_only=True)["threshold"]

        self.model = VisionLSTMAutoencoder(input_dim=NUM_FEATURES, hidden_dim=32, num_layers=2)
        self.model.load_state_dict(torch.load(model_path, weights_only=True))
        self.model.eval()

        self.seq_length = seq_length
        self.stride = stride

    def score_windows(self, windows: np.ndarray) -> np.ndarray:
        """(N, seq_len, 6) → 윈도우별 재구성 오차 (N,)"""
        x = torch.tensor(windows, dtype=torch.float32)
        x_norm = (x - self.mean) / (self.std + 1e-7)
        with torch.no_grad():
            recon = self.model(x_norm)
            errors = ((recon - x_norm) ** 2).mean(dim=(1, 2))
        return errors.numpy()

    def score_video(self, video_path: str) -> dict:
        """영상 1개 → 종합 이상 점수 + 판정"""
        series = extract_au_series(video_path, skip_frames=10)
        windows = build_sequences(series, seq_length=self.seq_length, stride=self.stride)
        if windows is None:
            return {"video_score": None, "is_anomaly": None, "n_windows": 0}

        errors = self.score_windows(windows)
        video_score = float(errors.mean())   # 영상 전체 대표 점수 = 윈도우 오차 평균
        return {
            "video_score": video_score,
            "is_anomaly": bool(video_score > self.threshold),
            "n_windows": len(errors),
        }


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/test.mp4"
    detector = VisionAnomalyDetector()
    result = detector.score_video(path)

    print(f"\n영상: {path}")
    print(f"  이상 점수: {result['video_score']}")
    print(f"  임계값   : {detector.threshold:.6f}")
    print(f"  판정     : {'이상(의심)' if result['is_anomaly'] else '정상'}")
    print(f"  분석 윈도우 수: {result['n_windows']}")
