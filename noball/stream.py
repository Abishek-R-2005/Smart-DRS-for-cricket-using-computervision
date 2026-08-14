import streamlit as st
import tempfile
import os
from main import process_video

st.set_page_config(layout="wide")
st.title("🏏 No Ball Detection System")

uploaded_file = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"])

if uploaded_file:

    # Save uploaded video
    temp_input_path = os.path.join(tempfile.gettempdir(), uploaded_file.name)

    with open(temp_input_path, "wb") as f:
        f.write(uploaded_file.read())

    st.subheader("📥 Input Video")
    st.video(temp_input_path)

    if st.button("🚀 Process Video"):

        with st.spinner("Processing..."):

            output_path = os.path.join(tempfile.gettempdir(), "output.mp4")

            process_video(temp_input_path, output_path)

        st.success("✅ Processing Completed!")

        st.subheader("📤 Output Video")

        # ✅ IMPORTANT FIX (READ AS BYTES)
        with open(output_path, "rb") as video_file:
            video_bytes = video_file.read()

        st.video(video_bytes)

        st.download_button(
            label="⬇️ Download Output",
            data=video_bytes,
            file_name="output.mp4",
            mime="video/mp4"
        )