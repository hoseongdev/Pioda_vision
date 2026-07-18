# Pioda_vision

**이음(以音) — 가족 소통 기반 치매 조기 징후 탐지 앱**의 영상(표정) 분석 모듈.

노인이 매일 가족에게 보내는 짧은 영상 일기 속 표정 변화를 비지도 학습으로 분석해, 인지 기능 저하와 관련된 안면 동결(hypomimia) 신호를 조용히 추적한다.

## 이음 프로젝트 전체 구조

이음은 팀원별로 역할을 나눈 풀스택 프로젝트이며, 이 레포는 그중 **영상 분석 파트**를 담당한다.

| 레포 | 역할 | 담당 |
|---|---|---|
| [ieum-android](https://github.com/2026-ieum-project/ieum-android) | Android 앱 (Kotlin) | 팀원 |
| [ieum-server](https://github.com/2026-ieum-project/ieum-server) | 음성 분석 + LSTM 추론 서버 (Flask/FastAPI) | 팀원 |
| **Pioda_vision (이 레포)** | 영상(표정) 분석 모듈 | 본인 |
| [Pioda-Project](https://github.com/Hottae0/Pioda-Project) | 음성+영상 통합/앙상블 버전 | 팀원 + 본인 |

## 문제 정의 및 설계 배경

프로젝트 초기에는 음성 + 키스트로크(타이핑 패턴) 두 채널을 분석하는 설계였으나, 노인 사용자에게 타이핑 자체가 진입장벽이 된다는 판단 아래 **음성 + 영상(표정)** 조합으로 피벗했다. 이 레포는 그중 영상 채널을 담당한다.

## 접근 방식

### 1. 표정 지표 추출 (`feature_extractor.py`)
- [Py-Feat](https://py-feat.org/)(MIT 라이선스)의 `Detectorv2`로 영상에서 프레임별 **Action Unit(AU)** 20종을 추출
- AU는 FACS(Facial Action Coding System) 기반 표정 근육 활성도 지표로, 파킨슨병·치매의 안면 동결(hypomimia) 연구에서 임상적으로 가장 널리 쓰이는 지표
- 온디바이스 경량화 제약을 두지 않기로 방향을 잡아(멘토링 피드백 반영), 무겁지만 정확한 Py-Feat을 서버 사이드에서 사용

### 2. 비지도 이상 탐지 (`model_train.py`)
- 정상 노인의 표정 움직임 패턴을 **LSTM 오토인코더**로 학습 (음성 모듈과 동일한 방식으로 설계해 인터페이스 일관성 확보)
- 정상 패턴을 잘 복원하지 못하는(재구성 오차가 큰) 입력을 이상(인지 저하 의심 신호)으로 판단
- Z-score 정규화 통계(`norm_stats.pt`)를 학습 시점에 저장해 추론 시에도 동일하게 적용
- 이상 탐지 임계값은 재구성 오차의 `mean + 2σ`로 설정 (`threshold.pt`)

### 3. 추론 (`inference.py`)
- 영상 1개를 받아 윈도우 단위로 재구성 오차를 계산하고, 평균을 영상 전체의 이상 점수로 사용
- `VisionAnomalyDetector` 클래스로 캡슐화해 음성 모듈과 동일한 인터페이스 유지 → 통합 레포([Pioda-Project](https://github.com/Hottae0/Pioda-Project))에서 음성 점수와 앙상블

### 4. 테스트 도구 (`record_test.py`)
- 웹캠으로 테스트 영상을 직접 녹화해 파이프라인 전체(추출 → 학습 → 추론)를 검증할 수 있는 스크립트
- 실제 녹화 영상으로 피처 추출부터 이상 탐지까지 전체 흐름 검증 완료

## 파일 구조

```
Pioda_vision/
├── feature_extractor.py     # 영상 → Py-Feat → AU 시계열 추출
├── model_train.py            # LSTM 오토인코더 학습
├── inference.py               # 이상 탐지 추론
├── record_test.py             # 웹캠 테스트 영상 녹화 도구
├── vision_autoencoder.pt      # 학습된 모델 가중치
├── norm_stats.pt              # 정규화 통계 (mean, std)
├── threshold.pt                # 이상 탐지 임계값
└── requirements.txt
```

## 기술적 의사결정

- **지도학습 대신 비지도 이상 탐지를 택한 이유**: 치매 여부가 라벨링된 대규모 얼굴 데이터를 구하기 어렵고, 개인마다 표정 baseline이 다르기 때문에 "정상 패턴에서 벗어난 정도"를 측정하는 방식이 더 현실적
- **AU(Action Unit) 기반 피처를 택한 이유**: 픽셀 자체보다 임상적으로 검증된 표정 근육 활성도 지표를 사용해 해석 가능성을 확보
- **음성 모듈과 동일한 LSTM 오토인코더 구조를 사용한 이유**: 인터페이스 일관성을 맞춰 팀원의 음성 모듈과 손쉽게 앙상블할 수 있도록 설계

## 향후 개선 방향

- `inference.py` 독스트링에 남아있는 구버전 피처 차원 표기(`(N, seq_len, 6)`) 정정 필요 (실제는 20)
- 실제 노인 사용자 대상 검증 데이터 확보 시 임계값 재조정 필요
