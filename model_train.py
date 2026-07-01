"""
model_train.py — 영상 안면 지표 기반 LSTM 오토인코더 학습

[개요]
    정상 노인의 영상일기 얼굴 데이터로 "정상 표정 움직임 패턴"을 학습한다.
    음성 모듈과 동일한 비지도 이상 탐지 방식 — 학습된 정상 패턴을 잘 복원하지
    못하는(재구성 오차가 큰) 입력을 '이상(안면 동결 등 인지 저하 의심)'으로 판단.

[입력 피처 20개]  (feature_extractor.extract_au_series)
    Py-Feat이 추출한 Action Unit(AU) 20종 활성도 (AU01~AU43).
    AU는 FACS 기반 표정 근육 단위. 치매의 안면 동결(hypomimia)은
    이 AU들의 '시간당 변동 폭'이 줄어드는 특징.

[정규화]  Z-score. 학습 데이터 전체의 피처별 mean/std로 (x-mean)/std.
          추론 시 동일 통계 필요 → norm_stats.pt 저장.

[산출물]
    vision_autoencoder.pt  학습된 모델 가중치
    norm_stats.pt          정규화 통계 (mean, std)
    threshold.pt           이상 탐지 임계값 (mean + 2σ)

[실행]  python model_train.py   (데이터: data/*.mp4)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from feature_extractor import build_dataset_from_dir, NUM_FEATURES


# ── 모델 정의 (음성 모듈과 동일 구조) ──────────────────────

class VisionLSTMAutoencoder(nn.Module):
    """
    LSTM 오토인코더.
      encoder      : 시퀀스(seq_len, 20) → 마지막 hidden state로 압축
      decoder_lstm : 압축 벡터를 seq_len 길이로 펼쳐 복원
      output_layer : hidden_dim → 원래 피처 차원(20)
    입력을 그대로 복원하도록 학습, 복원 오차(MSE)를 이상 점수로 사용.
    """
    def __init__(self, input_dim=NUM_FEATURES, hidden_dim=32, num_layers=2):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.decoder_lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        _, (hidden, _) = self.encoder(x)
        hidden_last = hidden[-1].unsqueeze(1).repeat(1, x.size(1), 1)
        decoded, _ = self.decoder_lstm(hidden_last)
        return self.output_layer(decoded)


if __name__ == "__main__":
    # 10fps 기준 30프레임=3초 윈도우. (테스트 영상 1개 검증 시엔 15/5로 줄여 사용)
    SEQ_LENGTH = 30
    STRIDE     = 15

    # ── 데이터 로드 ────────────────────────────────────────
    data = build_dataset_from_dir("data", seq_length=SEQ_LENGTH, stride=STRIDE, skip_frames=5)
    if data is None:
        raise RuntimeError("학습 데이터 없음. data/ 폴더에 영상을 넣으세요.")

    # Py-Feat detect()가 전역 gradient를 꺼둔 채로 반환하므로 학습 전 복구
    torch.set_grad_enabled(True)

    vision_data = torch.tensor(data, dtype=torch.float32)

    # ── 정규화 (Z-score) ───────────────────────────────────
    mean = vision_data.mean(dim=(0, 1), keepdim=True)
    std  = vision_data.std(dim=(0, 1), keepdim=True)
    data_norm = (vision_data - mean) / (std + 1e-7)
    torch.save({"mean": mean, "std": std}, "norm_stats.pt")
    print("[train] 정규화 통계 저장 완료")

    # ── 학습 ───────────────────────────────────────────────
    dataset    = TensorDataset(data_norm, data_norm)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model     = VisionLSTMAutoencoder(input_dim=NUM_FEATURES, hidden_dim=32, num_layers=2)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    EPOCHS = 100
    print(f"[train] 학습 시작 ({EPOCHS} epochs, {len(dataloader)} batches/epoch)")

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch [{epoch+1}/{EPOCHS}]  Loss: {epoch_loss/len(dataloader):.6f}")

    # ── 이상 탐지 임계값 산출 ──────────────────────────────
    model.eval()
    with torch.no_grad():
        all_errors = []
        for (batch_x,) in DataLoader(TensorDataset(data_norm), batch_size=256):
            recon = model(batch_x)
            all_errors.append(((recon - batch_x) ** 2).mean(dim=(1, 2)))
        all_errors = torch.cat(all_errors)
        t_mean = all_errors.mean().item()
        t_std  = all_errors.std().item()
        threshold = t_mean + 2 * t_std

    print(f"\n[train] 재구성 오차 — mean: {t_mean:.6f}, std: {t_std:.6f}")
    print(f"[train] 이상 탐지 임계값 (mean + 2σ): {threshold:.6f}")

    # ── 저장 ───────────────────────────────────────────────
    torch.save(model.state_dict(), "vision_autoencoder.pt")
    torch.save({"threshold": threshold, "mean": t_mean, "std": t_std}, "threshold.pt")
    print("\n[train] 저장 완료: vision_autoencoder.pt / norm_stats.pt / threshold.pt")
