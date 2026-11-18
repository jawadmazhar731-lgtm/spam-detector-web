# SMS Spam Detector — Web Version (Streamlit Cloud frontend)
# Uses the SAME backend as the Flask app: Jawad's original code
# (spam_detector.py), only imported — no logic rewritten.

import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st  # noqa: E402

import spam_detector  # noqa: E402  (trains the model on import)

st.set_page_config(page_title="SMS Spam Detector", layout="centered")
st.title("SMS Spam Detector")
st.write("Type a message and find out whether it is spam or not.")

msg = st.text_area("Message to check", height=120)
if st.button("Check Message"):
    if msg.strip():
        result = spam_detector.predict_message(msg)
        if result == "SPAM":
            st.error("SPAM — This message looks like spam.")
        else:
            st.success("HAM — Not Spam. This message looks normal.")
    else:
        st.warning("Please enter a message first.")
