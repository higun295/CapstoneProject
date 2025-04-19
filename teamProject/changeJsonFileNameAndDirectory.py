import os
import shutil
import re

# 루트 폴더
base_dir = r"D:\Data\Outputs"

# 하위 폴더 순회
for folder_name in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder_name)
    if not os.path.isdir(folder_path):
        continue

    json_file = os.path.join(folder_path, "alphapose-results.json")
    if not os.path.exists(json_file):
        continue

    # 새로운 이름: SubjectXActivityY.json 추출
    match = re.search(r"(Subject\d+Activity\d+)", folder_name)
    if not match:
        continue

    new_filename = match.group(1) + ".json"
    new_filepath = os.path.join(base_dir, new_filename)

    # 이동 (또는 복사)
    shutil.move(json_file, new_filepath)
    print(f"✅ {json_file} → {new_filepath}")
