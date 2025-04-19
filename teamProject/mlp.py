import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ★ 여기에 FinalData 폴더 경로를 설정해줘
json_dir = r"D:\Data\FinalData"

X_all = []
y_all = []

# 1. 모든 JSON 파일 순회
for file in os.listdir(json_dir):
    if file.endswith(".json"):
        with open(os.path.join(json_dir, file), "r", encoding="utf-8") as f:
            data = json.load(f)
            for item in data:
                keypoints = item.get("keypoints", [])
                label = item.get("label", None)
                if not keypoints or label is None:
                    continue
                # 2. 라벨 매핑 (0: 1~5, 1: 6~11, 20)
                if label in [1, 2, 3, 4, 5]:
                    mapped_label = 0
                elif label in [6, 7, 8, 9, 10, 11, 20]:
                    mapped_label = 1
                else:
                    continue
                X_all.append(keypoints)
                y_all.append(mapped_label)

# 3. numpy 변환
X = np.array(X_all)
y = np.array(y_all)

# 4. 학습/테스트 분리 (7:3)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)

# 5. MLPClassifier 정의 (그림의 하이퍼파라미터 기반)
clf = MLPClassifier(
    hidden_layer_sizes=(100,),
    activation='relu',
    solver='sgd',
    alpha=1e-8,
    batch_size=min(200, len(X_train)),
    learning_rate_init=0.001,
    max_iter=10,
    shuffle=True,
    tol=0.0001,
    momentum=0.9,
    n_iter_no_change=10,
    verbose=True
)

# 6. 학습
clf.fit(X_train, y_train)

# 7. 예측
y_pred = clf.predict(X_test)

# 8. 평가 지표 계산
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
specificity = tn / (tn + fp)

# 9. 결과 출력
print("\n🧪 [MLP Classification 평가 결과]")
print(f"Accuracy     : {accuracy:.4f}")
print(f"Precision    : {precision:.4f}")
print(f"Recall       : {recall:.4f}")
print(f"Specificity  : {specificity:.4f}")
print(f"F1 Score     : {f1:.4f}")
