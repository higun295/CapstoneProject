import zipfile
import os

# 압축 파일이 들어 있는 루트 폴더
camera_root = "D:/Data/DaHyeon/Zipped"

# 압축 해제 결과를 저장할 루트 폴더
extract_root = "D:/Data/DaHyeon/Camera"

# 모든 Subject 폴더 순회
for subject_folder in os.listdir(camera_root):
    subject_path = os.path.join(camera_root, subject_folder)

    if not os.path.isdir(subject_path):
        continue

    # Subject 폴더 안의 zip 파일 순회
    for file in os.listdir(subject_path):
        if file.lower().endswith(".zip"):
            zip_path = os.path.join(subject_path, file)

            # 압축 해제 폴더명 = zip 이름과 동일
            folder_name = os.path.splitext(file)[0]
            extract_path = os.path.join(extract_root, subject_folder, folder_name)

            os.makedirs(extract_path, exist_ok=True)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)

            print(f"✅ {zip_path} → {extract_path} 압축 해제 완료")
