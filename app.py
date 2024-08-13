import streamlit as st
import numpy as np
import time
import pickle
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()

    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("Fake News Detection")

input_sms = st.text_area("Enter the News")

col1, col2 = st.columns(2)

def prediction(real, fake):
    progress_placeholder = st.empty()
    temp = st.text("Calculating....")
    temp.markdown('<div style="font-weight: bold; font-size: 35px;">Calculating....</div>', unsafe_allow_html=True)

    for i in range(1, real):
        html_code = f"""
        <style>
            .custom-bar-container {{
                width: 100%;
                height: 30px;
                border: 1px solid #ccc;
                border-radius: 5px;
                overflow: hidden;
            }}
            .custom-bar-green {{
                background-color: green;
                height: 100%;
                float: left;
                transition: width 0.5s; /* Add transition for smooth animation */
            }}
            .custom-bar-red {{
                background-color: red;
                height: 100%;
                float: left;
                transition: width 0.5s; 
            }}
        </style>
        <div class="custom-bar-container">
            <div class="custom-bar-green" style="width: {i}%"></div>
            <div class="custom-bar-red" style="width: {100 - i}%"></div>
        </div>
        """
        progress_placeholder.write(html_code, unsafe_allow_html=True)
        time.sleep(0.1)

    temp.empty()
    html = f'<div style="display: flex; justify-content: space-between;">'
    html += f'<div style="font-weight: bold; font-size: 25px;">Real - {real}%</div>'
    html += f'<div style="font-weight: bold; font-size: 25px;">Fake - {fake}%</div>'
    html += f'</div>'
    st.markdown(html, unsafe_allow_html=True)
    
if col1.button('Predict'):
    if not input_sms:
        st.warning("Please enter a News first!!!")
    else:
        transformed_sms = transform_text(input_sms)
        vector_input = tfidf.transform([transformed_sms])
        # res = model.predict(vector_input)[0]
        res=model.decision_function(vector_input)[0]
        st.header(str(res))
        result = 1 / (1 + np.exp(-res))
        # st.header("Real"+"-"+str(round(result[0]*100))+"%")
        # st.header("Fake"+"-"+str(round(100-result[0]*100))+"%")

        # real = round(result[0] * 100)
        real=round(result * 100)
        fake = round(100 - real)

        st.header("Real"+"-"+str(round(result*100))+"%")
        st.header("Fake"+"-"+str(round(100-result*100))+"%")

        prediction(real, fake)

    # Place button in the second column
if col2.button('Clear Result'):
    st.header(" ")

# if st.button('Predict'):

#     # 1. preprocess
#     transformed_sms = transform_text(input_sms)
#     # 2. vectorize
#     vector_input = tfidf.transform([transformed_sms])
#     # 3. predict
#     result = model.predict(vector_input)[0]
#     # 4. Display
#     if result == 0:
#         st.header("Fake")
#     else:
#         st.header("Real")
