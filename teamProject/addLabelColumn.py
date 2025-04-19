import os
import json
import pandas as pd

# 경로 설정
csv_dir = r"D:\Data\csv"
json_dir = r"D:\Data\json"
output_dir = r"D:\Data\FinalData"
os.makedirs(output_dir, exist_ok=True)

# _파일만 필터링
csv_files = [f for f in os.listdir(csv_dir) if f.startswith("_") and f.endswith(".csv")]

for csv_file in csv_files:
    # 파일명에서 SubjectXActivityY 추출
    base_name = csv_file.lstrip("_").replace("Trial1", "").replace(".csv", "")
    json_file = base_name + ".json"

    csv_path = os.path.join(csv_dir, csv_file)
    json_path = os.path.join(json_dir, json_file)
    output_path = os.path.join(output_dir, base_name + ".json")

    # 파일 존재 확인
    if not os.path.exists(json_path):
        print(f"⚠️ JSON 파일 없음: {json_path}")
        continue

    # CSV 로드
    df = pd.read_csv(csv_path)
    df["image_id"] = df["TimeStamps"].str.replace(":", "_") + ".png"

    label_map = dict(zip(df["image_id"], df["Label"]))

    # JSON 로드
    with open(json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    # JSON에 label 추가
    for item in json_data:
        img_id = item.get("image_id")
        if img_id in label_map:
            item["label"] = int(label_map[img_id])

    # 저장
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)

    print(f"✅ 병합 완료: {output_path}")
