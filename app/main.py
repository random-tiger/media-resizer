# app/main.py

import streamlit as st
from video_resizer import video_uploader

def main():
    st.title("Video Resizer")
    st.write("Upload a video file to resize.")
    
    operation_mode = st.sidebar.selectbox(
        "Select Operation Mode", 
        ["Video Resizer"]
    )
    
    if operation_mode == "Video Resizer":
        video_uploader()
        
if __name__ == "__main__":
    main()
