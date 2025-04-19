import os
import pandas as pd

csv_dir = r"D:\Data\csv"

for file in os.listdir(csv_dir):
    if not (file.endswith(".csv") and file.startswith("Subject")):
        continue

    input_path = os.path.join(csv_dir, file)
    output_path = os.path.join(csv_dir, f"_{file}")

    # 1. 줄 단위로 읽기
    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # 2. 헤더에 , 추가 + 두 번째 줄 삭제
    if len(lines) > 1:
        if not lines[0].strip().endswith(","):
            lines[0] = lines[0].strip() + ",\n"  # 헤더 수정
        del lines[1]  # 2번째 줄 삭제

    # 3. 임시 저장
    temp_path = os.path.join(csv_dir, "_temp.csv")
    with open(temp_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    # 4. pandas로 읽기
    df = pd.read_csv(temp_path)

    # 5. TimeStamps + 마지막 컬럼만 추출
    if "TimeStamps" not in df.columns or len(df.columns) < 2:
        print(f"⚠️ TimeStamps 없음 or 컬럼 부족: {file}")
        continue

    label_col = df.columns[-1]
    df = df[["TimeStamps", label_col]]
    df.rename(columns={label_col: "Label"}, inplace=True)

    # 6. 저장 (덮어쓰기 or 생성)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"✅ 처리 완료: {output_path}")

# 7. 임시 파일 제거
os.remove(temp_path)
