# 이하 내용을 app.py에 write
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from model import predict  # 이전에 변환한 Keras 버전 predict 함수 불러오기

# 사이드바 UI
st.sidebar.title("이미지 인식 앱")
st.sidebar.write("Keras CNN 모델을 사용해서 Fashion-MNIST 이미지를 판정합니다.")
st.sidebar.write("")

# 이미지 입력 소스 선택
img_source = st.sidebar.radio("이미지 소스를 선택해 주세요.",
                              ("이미지를 업로드", "카메라로 촬영"))

if img_source == "이미지를 업로드":
    img_file = st.sidebar.file_uploader("이미지를 선택해 주세요.", type=["png", "jpg", "jpeg"])
elif img_source == "카메라로 촬영":
    img_file = st.camera_input("카메라로 촬영")

# 이미지가 입력되었을 때 실행
if img_file is not None:
    with st.spinner("측정 중..."):
        img = Image.open(img_file)
        st.image(img, caption="대상 이미지", width=480)
        st.write("")

        # 예측 실행
        results = predict(img)

        # 결과 표시
        st.subheader("판정 결과")
        n_top = 3  # 확률이 높은 순으로 3위까지 출력
        for result in results[:n_top]:
            st.write(f"{round(result[2]*100, 2)}%의 확률로 {result[0]} 입니다.")

        # 원형 차트 표시
        pie_labels = [result[1] for result in results[:n_top]]
        pie_labels.append("others")  # 기타
        pie_probs = [result[2] for result in results[:n_top]]
        pie_probs.append(sum([result[2] for result in results[n_top:]]))  # 기타 비율

        fig, ax = plt.subplots()
        wedgeprops = {"width": 0.3, "edgecolor": "white"}
        textprops = {"fontsize": 6}
        ax.pie(pie_probs, labels=pie_labels, counterclock=False, startangle=90,
               textprops=textprops, autopct="%.2f", wedgeprops=wedgeprops)
        st.pyplot(fig)

# 사이드바 안내문
st.sidebar.write("")
st.sidebar.write("")
st.sidebar.caption("""
이 앱은「Fashion-MNIST」데이터셋을 기반으로 훈련된 Keras CNN 모델을 사용합니다.\n
Copyright (c) 2017 Zalando SE\n
Released under the MIT license\n
https://github.com/zalandoresearch/fashion-mnist#license
""")
