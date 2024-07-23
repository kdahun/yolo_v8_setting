## GPU 버전 확인
cmd -> nvidia-smi

## GPU 사용 가능한 pytorch 환경 구축
1) GPU 드라이버 설치
2) pytorch 홈페이지에서 원하는 CUDA버전의 pytorch 버전 다운로드
3) pytorch 버전에 맞는 CUDA 설치
4) CUDA 버전에 맞는 cudnn 설치
5) cmd -> nvcc --version를 사용해 설치 확인


# install
 : Python >= 3.8 버전과 Pytorch >= 1.8 환경에서 reqirements가 포함된 ultralytics 패키지 다운로드
```
pip install ultralytics
```

# Quick start
```
from ultralytics import YOLO
import cv2

model = YOLO('yolov8s.pt')

result = model.predict("./zidane.jpg, save = True, conf = 0.5")

```

### pt 파일
: pt 파일은 Pytorch에서 사용하는 모델 파일 형식이다. PyTorch는 Facebook에서 개발한 오픈 소스 머신러닝 라이브러리로, 주로 딥 러닝 연구와 애플리케이션에 사용된다. pt 파일에는 학습된 모델의 가중치(weight), 구조(graph), 매개변수(parameters) 등이 저장되어 있다.
yolo 모델의 pt 파일은 사전 학습된 가중치가 포함된 모델 파일로, 이를 통해 모델을 처음부터 다시 학습시키지 않고도 객체 탐지 작업을 수행할 수 있다.

* YOLO('yolov8s.pt')
  - yolov8s.pt는 사전 학습된 모델 가중치 파일의 경로

* model.predict("./zidane.jpg, save = True, conf = 0.5")
  - source : 예측할 이미지나 비디오의 경로
  - save : 예측 결과를 저장할지 여부
  - conf : 예측의 신뢰도 임계값. 이 값보다 낮은 신뢰도의 예측은 무시된다.
 
### result 객체 속성
```
# 첫 번째 결과 가져오기
result = result[0]
print("Labels:", result.names)
print("Boxes:", result.boxes)
print("Masks:", result.masks)
print("Keypoints:", result.keypoints)
print("Original Image Shape:", result.orig_img.shape)
```
1. Labels(클래스 이름) : 클래스 ID와 이름의 매핑을 보여준다.
2. Boxes(탐지된 객체의 경계 상자 정보) : 각 객체의 경계 상자 좌표, 클래스 ID, 신뢰도 점수를 포함한다.
3. Masks(객체의 세그멘테이션 마스크)
4. Keypoints(객체의 키포인트 정보)
5. Original Image(원본 이미지 배열) : 원본 이미지의 크기와 채널 수를 보여준다.

## 이미지에 경계 상자 그리기
```
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# YOLO 모델 불러오기
model = YOLO('yolov8s.pt')

# 이미지 예측
results = model.predict("./zidane.jpg", save=True, conf=0.5)

# 첫 번째 결과 가져오기
result = results[0]

# 원본 이미지 가져오기
original_img = result.orig_img

# 탐지된 객체의 경계 상자 그리기
for box in result.boxes:
    x1, y1, x2, y2 = box.xyxy[0]  # 경계 상자 좌표
    conf = box.conf[0]  # 신뢰도
    cls = box.cls[0]  # 클래스 ID
    label = result.names[int(cls)]  # 클래스 이름

    # 경계 상자 그리기
    cv2.rectangle(original_img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
    cv2.putText(original_img, f'{label} {conf:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

# 결과 이미지 표시
plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
plt.show()


```

![image](https://github.com/user-attachments/assets/2aecfd70-7ca1-420e-9dd2-3e872fcce282)

---

# 02.person_detect

```
from ultralytics import YOLO
import cv2
model = YOLO("yolov8n.pt") # 원하는 크기 모델 입력(n ~ x)

# 모델을 사용해 이미지 예측
result = model.predict(source = "./frame/notebook/*.png", conf=0.5)

```

사전 학습된 80개의 클래스를 포함하여 COCO에서 학습된 yolyv8n.pt 모델을 사용해 result에 저장

```
import os
import shutil

# 폴더 생성
person_predict_folder = "person_predict_notebook"
if not os.path.exists(person_predict_folder):
    os.makedirs(person_predict_folder)

person_results=[]
# 결과 처리
for re in result:
    # 각 예측 결과를 반복하여 사람 객체(cls==0)이 있는지 확인
    has_person = any(int(cls) == 0 for cls in re.boxes.cls)

    # 사람 객체가 있는 경우, 해당 이미지를 person_predict 폴더에 복사하여 저장
    if has_person:
        person_results.append(re)
        # 원본 이미지 경로 가져오기
        original_image_path = re.path
        # 파일명만 추출
        file_name = os.path.basename(original_image_path)
        # 새 저장 경로 지정
        new_path = os.path.join(person_predict_folder, file_name)
        # 이미지 복사 및 저장
        shutil.copy(original_image_path, new_path)

```


