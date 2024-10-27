# app/main.py

import streamlit as st
from .video_resizer import video_uploader
from .subtitle_creator import subtitle_creation_mode
from .scene_search import scene_search_mode

def main():
    st.title("Media Studio")
    st.write("Upload a video file to resize, convert, add subtitles, or search scenes based on prompts.")
    
    operation_mode = st.sidebar.selectbox(
        "Select Operation Mode", 
        ["Video Resizer", "Subtitle Creator", "Scene Search"]
    )
    
    if operation_mode == "Video Resizer":
        video_uploader()
    elif operation_mode == "Subtitle Creator":
        subtitle_creation_mode()
    elif operation_mode == "Scene Search":
        scene_search_mode()

if __name__ == "__main__":
    main()
