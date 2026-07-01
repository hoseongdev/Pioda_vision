"""녹화된 영상 점검 — 프레임 읽기 + 얼굴 검출 여부 + 샘플 프레임 저장"""
import cv2
import numpy as np
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

path = "data/test.mp4"
cap = cv2.VideoCapture(path)
print("열림:", cap.isOpened())
print("fps:", cap.get(cv2.CAP_PROP_FPS))
print("프레임수:", cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("해상도:", cap.get(cv2.CAP_PROP_FRAME_WIDTH), "x", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

frames = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)
cap.release()
print("실제 읽힌 프레임:", len(frames))

if frames:
    mid = frames[len(frames) // 2]
    print("중간 프레임 밝기(평균):", float(np.mean(mid)))
    cv2.imwrite("data/sample_frame.jpg", mid)
    print("→ data/sample_frame.jpg 저장 (얼굴 보이는지 확인용)")

    # 검출 신뢰도 낮춰서 재시도
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                               min_detection_confidence=0.2) as fm:
        hit = 0
        for f in frames[::5]:
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            if fm.process(rgb).multi_face_landmarks:
                hit += 1
        print(f"낮은 신뢰도(0.2)로 검출된 프레임: {hit} / {len(frames[::5])}")
