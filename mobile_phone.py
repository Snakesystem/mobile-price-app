import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(
        page_title="MBAPP",
        page_icon="chart_with_upwards_trend",
        layout="wide",
    )

def Model():
    st.subheader('Mobile Phone Prediction')

    st.sidebar.header('Dokumentation')
    st.sidebar.write("Dikembangkan oleh:")
    st.sidebar.subheader('Feri Irawansyah')
    st.sidebar.write("[Artikel penelitian](https://github.com/Snakesystem/mobile-price-analisyst)")
    st.sidebar.write("Profile Development")
    
    st.sidebar.write("""
                     [Feri Irawansyah](https://www.linkedin.com/in/feri-irawansyah-154504216/) & 
                     [Achmad Reihan](https://www.linkedin.com/in/achmad-raihan-477ab61b9/) 
                     """)
    

    def user_input_features():
        battery_power = st.slider('Battery Power ( mAh )', 500, 2000)
        int_memory = st.slider('Internal Memory ( GB )', 2, 64)
        ram = st.slider('Ram ( MB )', 256, 4000)
        px_width = st.slider('Width Resolution ( PX )', 500, 2000)
        px_height = st.slider('Height Resolution ( PX )', 1, 2000)
        data = {'battery_power': battery_power,
                'int_memory': int_memory,
                'ram': ram,
                'px_width': px_width,
                'px_height': px_height,
                }
        features = pd.DataFrame(data, index=[0])
        return features

    input_df = user_input_features()

    # Load Model
    load_model = pickle.load(open('mobile_rf_performance.pkl', 'rb'))

    # Apply model to make predictions
    prediction = load_model.predict(input_df)
    prediction_proba = load_model.predict_proba(input_df)

    st.subheader('Result Prediction')

    if prediction == 0:
        st.error("Your Phone Is Low Quality")
    elif prediction == 1:
        st.warning('Your Phone Is Normal Quality')
    elif prediction == 2:
        st.info("Your Phone Is Medium Quality")
    else:
        st.success("Your Phone Is Hight Quality")

    st.subheader('Prediction Probability',)
    quality = (prediction_proba)
    proba = quality.reshape(-1)
    labels = 'Low', 'Normal', ' Medium', 'Hight'

    st.write(proba)
    chart_data = proba.reshape(1, -1)
    plot_data = pd.DataFrame(proba, index=labels)
    st.bar_chart(plot_data)


if __name__ == '__main__':
    Model()
