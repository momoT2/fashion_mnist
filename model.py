# 이하 내용을 model.py에 write
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image

# 클래스 정의
classes_kr = ["티셔츠/톱", "바지", "풀오버", "드레스", "코트",
              "샌들", "와이셔츠", "스니커즈", "가방", "앵클부츠"]
classes_en = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
              "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

n_class = len(classes_kr)
img_size = 28

# 저장된 모델 불러오기
model = keras.models.load_model("model_cnn_final.keras")
# model = build_model()
# model.load_weights("model_cnn.weights.h5")

# 예측 함수
def predict(img):
    # 흑백 변환 및 크기 조정
    img = img.convert("L") # 8비트 그레이스케일 모드
    img = img.resize((img_size, img_size))

    # NumPy 배열로 변환 후 정규화
    img_array = np.array(img).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=-1)  # 채널 추가
    img_array = np.expand_dims(img_array, axis=0)   # 배치 차원 추가 (1,28,28,1)

    # 예측
    y_prob = model.predict(img_array, verbose=0)[0]  # shape: (10,)
    sorted_indices = np.argsort(y_prob)[::-1]        # 내림차순 정렬

    return [(classes_kr[idx], classes_en[idx], float(y_prob[idx]))
            for idx in sorted_indices]
