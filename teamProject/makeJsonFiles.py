import os
import subprocess

# AlphaPose 설치 경로 (가상환경 기준 루트)
ALPHAPOSE_DIR = r"C:\Users\HEEKWON\AlphaPose"
ALPHAPOSE_SCRIPT = os.path.join(ALPHAPOSE_DIR, "scripts", "demo_inference.py")
POSE_CFG = os.path.join(ALPHAPOSE_DIR, "configs", "coco", "resnet", "256x192_res50_lr1e-3_1x.yaml")
POSE_MODEL = os.path.join(ALPHAPOSE_DIR, "pretrained_models", "fast_res50_256x192.pth")
POSE_BATCH = "32"

# 입력 및 출력 루트 경로
input_root = r"D:\Data\DaHyeon\Camera"
output_root = r"D:\Data\DaHyeon\outputs"

# Subject 폴더 순회
for subject in os.listdir(input_root):
    subject_path = os.path.join(input_root, subject)
    if not os.path.isdir(subject_path):
        continue

    for activity_folder in os.listdir(subject_path):
        activity_path = os.path.join(subject_path, activity_folder)
        if not os.path.isdir(activity_path):
            continue

        # 출력 경로 지정
        output_path = os.path.join(output_root, activity_folder)
        os.makedirs(output_path, exist_ok=True)

        # 명령어 구성
        command = [
            "python", ALPHAPOSE_SCRIPT,
            "--cfg", POSE_CFG,
            "--checkpoint", POSE_MODEL,
            "--indir", activity_path,
            "--outdir", output_path,
            "--posebatch", POSE_BATCH
        ]

        print(f"🚀 AlphaPose 실행 중: {activity_path}")
        subprocess.run(command, cwd=ALPHAPOSE_DIR)
