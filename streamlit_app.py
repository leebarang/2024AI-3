# 분류 결과 + 이미지 + 텍스트와 함께 분류 결과에 따라 다른 출력 보여주기
import streamlit as st
from fastai.vision.all import *
from PIL import Image
import gdown
from extract_yt_url import Extract_yt_url

# Google Drive 파일 ID
file_id = '1Yhwkt1kuzFaCubaukRlbi54IM2FJSkMF'

# Google Drive에서 파일 다운로드 함수
@st.cache(allow_output_mutation=True)
def load_model_from_drive(file_id):
    url = f'https://drive.google.com/uc?id={file_id}'
    output = 'model.pkl'
    gdown.download(url, output, quiet=False)

    # Fastai 모델 로드
    learner = load_learner(output)
    return learner

def display_left_content(image, prediction, probs, labels):
    st.write("### 예측 결과")
    if image is not None:
        st.image(image, caption="업로드된 이미지", use_column_width=True)
    st.write(f"예측된 클래스: {prediction}")
    st.markdown("<h4>예측 확률</h4>", unsafe_allow_html=True)
    for label, prob in zip(labels, probs):
        st.markdown(f"""
            <div style="background-color: #f0f0f0; border-radius: 5px; padding: 5px; margin: 5px 0;">
                <strong style="color: #333;">{label}:</strong>
                <div style="background-color: #d3d3d3; border-radius: 5px; width: 100%; padding: 2px;">
                    <div style="background-color: #4CAF50; width: {prob*100}%; padding: 5px 0; border-radius: 5px; text-align: center; color: white;">
                        {prob:.4f}
                    </div>
                </div>
        """, unsafe_allow_html=True)

search_dict = {0:"유튜브 검색 링크", 1:"위키 백과", 2:"먹이 구매 (쿠팡)"}
search_results = ['https://www.youtube.com/results?search_query=', 'https://ko.wikipedia.org/wiki/', 'https://www.coupang.com/np/search?component=&q=']

def display_right_content(prediction):
    st.write("### 관련 콘텐츠")
    cols = st.columns(3)
    
    # 1st Row - pop-up botton
    for i in range(3):
        with cols[i]:
            if st.button(f"{prediction} {search_dict[i]}"):
                if i == 2: # 먹이 검색
                    st.markdown(f'{search_results[i]}{prediction}+먹이', unsafe_allow_html=True)
                st.markdown(f'{search_results[i]}{prediction}', unsafe_allow_html=True)

    # 2nd Row - YouTube Videos
    for i in range(3):
        with cols[i]:
            st.video(Extract_yt_url(prediction,3).to_list()[i])
            st.caption(f"유튜브: {prediction}")

    st.write("### 직접 유튜브 영상 찾기")
    text_input = st.text_input("텍스트 입력", prediction)
    if text_input:
        st.write(f"입력된 텍스트: {text_input}")
        st.video(Extract_yt_url(text_input).to_string())
        st.caption(f"유튜브: {text_input}")

# 모델 로드
st.write("모델을 로드 중입니다. 잠시만 기다려주세요...")
learner = load_model_from_drive(file_id)
st.success("모델이 성공적으로 로드되었습니다!")

labels = learner.dls.vocab

# 스타일링을 통해 페이지 마진 줄이기
st.markdown("""
    <style>
    .reportview-container .main .block-container {
        max-width: 90%;
        padding-top: 1rem;
        padding-right: 1rem;
        padding-left: 1rem;
        padding-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# 레이아웃 설정
left_column, right_column = st.columns([1, 2])  # 왼쪽과 오른쪽의 비율 조정

# 파일 업로드 컴포넌트 (jpg, png, jpeg, webp, tiff 지원)
uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["jpg", "png", "jpeg", "webp", "tiff"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img = PILImage.create(uploaded_file)
    prediction, _, probs = learner.predict(img)

    with left_column:
        display_left_content(image, prediction, probs, labels)

    with right_column:
        # 분류 결과에 따른 콘텐츠 선택
        display_right_content(prediction)

