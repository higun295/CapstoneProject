import os
import shutil
import re

# 입력 루트 (원본 CSV들이 있는 폴더들)
base_dir = r"D:\Data\Dataset"

# Subject1 ~ Subject17 순회
for subject_folder in os.listdir(base_dir):
    subject_path = os.path.join(base_dir, subject_folder)
    if not os.path.isdir(subject_path):
        continue

    # 폴더 내 CSV 파일 탐색
    for file in os.listdir(subject_path):
        if not file.endswith(".csv"):
            continue

        old_path = os.path.join(subject_path, file)

        # 새로운 파일명: SubjectXActivityY.csv
        match = re.search(r"(Subject\d+Activity\d+)", file)
        if not match:
            continue

        new_filename = match.group(1) + ".csv"
        new_path = os.path.join(base_dir, new_filename)

        # 파일 이동
        shutil.copyfile(old_path, new_path)
        print(f"✅ {old_path} → {new_path}")
