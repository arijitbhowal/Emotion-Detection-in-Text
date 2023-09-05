# Core Pkgs
import streamlit as st
import altair as alt
import plotly.express as px

# EDA Pkgs
import pandas as pd
import numpy as np
from datetime import datetime

# Utils
import joblib

pipe_lr = joblib.load(open("models/emotion_classifier_pipe_lr.pkl", "rb"))


def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]


def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results


# Main Application
def main():
    st.title("Emotion Classifier App")
    tab1, tab2 = st.tabs(["Home", "About"])

    with tab1:
        st.subheader("Emotion in Text Classifier")
        with st.form("my_form"):
            txt = st.text_area("Enter the text here")
            # Every form must have a submit button.
            submitted = st.form_submit_button("Submit")
            prediction = predict_emotions(txt)
            probability = get_prediction_proba(txt)

            if submitted:
                col1, col2 = st.columns(2)

                with col1:
                    st.success("Original Text")
                    st.write(txt)
                    st.success("Prediction")
                    st.write(prediction)
                    st.write("Confidence: {:.4f}".format(np.max(probability)))

                with col2:
                    st.success("Prediction Probability")
                    proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
                    st.write(proba_df.T)  # Displaying the DataFrame
                    prob_df_transpose = proba_df.T.reset_index()
                    prob_df_transpose.columns = ["emotions", "probability"]
                    fig = (
                        alt.Chart(prob_df_transpose)
                        .mark_bar()
                        .encode(x="emotions", y="probability")
                    )
                    st.altair_chart(fig, use_container_width=True)
if __name__ == "__main__":
    main()
