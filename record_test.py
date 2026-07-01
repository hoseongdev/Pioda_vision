"""
record_test.py — 웹캠으로 테스트 영상 녹화 (피처 추출 검증용)

사용법:
    python record_test.py            # 10초 녹화 → data/test.mp4
    python record_test.py 15 out.mp4 # 15초 녹화 → data/out.mp4

녹화 중 얼굴을 카메라에 비추고, 중간에 한 번 눈을 깜빡이거나
표정을 바꿔보면 피처 값 변화를 확인하기 좋다.
"""

import cv2
import os
import sys
import time

DURATION = int(sys.argv[1]) if len(sys.argv) > 1 else 10
OUT_NAME = sys.argv[2] if len(sys.argv) > 2 else "test.mp4"

os.makedirs("data", exist_ok=True)
out_path = os.path.join("data", OUT_NAME)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("웹캠을 열 수 없음. 카메라 권한을 확인하세요.")

fps = 20
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

# 카메라 노출/화이트밸런스 안정화 — 시작 전 2초간 프레임 버림
print("[record] 카메라 워밍업 중... (2초)")
warmup_end = time.time() + 2.0
while time.time() < warmup_end:
    cap.read()

print(f"[record] {DURATION}초 녹화 시작... (얼굴을 카메라에 비추세요)")
start = time.time()
frame_count = 0

while time.time() - start < DURATION:
    ret, frame = cap.read()
    if not ret:
        break
    writer.write(frame)
    frame_count += 1
    remaining = DURATION - (time.time() - start)
    print(f"\r  녹화 중... {remaining:4.1f}초 남음", end="")

cap.release()
writer.release()
print(f"\n[record] 완료: {out_path} ({frame_count} 프레임)")
