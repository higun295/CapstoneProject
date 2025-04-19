import os
import json
from collections import defaultdict

# JSON 파일이 들어있는 폴더
json_dir = r"D:\Data\json"

# 폴더 내 모든 JSON 파일 순회
for filename in os.listdir(json_dir):
    if not filename.endswith(".json"):
        continue

    file_path = os.path.join(json_dir, filename)

    # JSON 로드
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # image_id 기준으로 그룹핑
    grouped = defaultdict(list)
    for item in data:
        grouped[item["image_id"]].append(item)

    # score 기준으로 최고 항목만 추출
    filtered_data = []
    for image_id, items in grouped.items():
        if len(items) == 1:
            filtered_data.append(items[0])
        else:
            best = max(items, key=lambda x: x.get("score", 0))
            filtered_data.append(best)

    # 덮어쓰기 저장
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(filtered_data, f, indent=2)

    print(f"✅ 필터링 완료: {filename} ({len(data)} → {len(filtered_data)} 개)")
