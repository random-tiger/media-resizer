import streamlit as st
from PIL import Image
import moviepy.editor as mp
import tempfile
import os
from streamlit_cropper import st_cropper

def main():
    st.title("Media Resizer, Cropper, and Converter")
    st.write("Upload an image or video file to resize, crop, and convert it to different formats.")

    media_type = st.sidebar.selectbox("Select Media Type", ["Image", "Video"])

    if media_type == "Image":
        image_uploader()
    else:
        video_uploader()

def image_uploader():
    uploaded_image = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png", "bmp", "gif"])
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.write("### Choose Operation")
        operation = st.selectbox("Operation", ["Resize", "Crop"])

        if operation == "Resize":
            resize_image(image)
        else:
            crop_image(image)

def resize_image(image):
    # [Your existing resize_image code]
    # Ensure you have the updated version that fixes previous issues.

def crop_image(image):
    # [Your existing crop_image code]

def video_uploader():
    uploaded_video = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov", "mkv"])
    if uploaded_video is not None:
        # Save uploaded video to a temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_video.read())
        tfile.flush()
        st.video(tfile.name)

        st.write("### Resize Options")
        resolution = st.selectbox("Resolution", ["1920x1080", "1280x720", "854x480", "640x360"])
        output_format = st.selectbox("Output Format", ["mp4", "avi", "mov", "mkv"])

        if st.button("Resize and Convert Video"):
            try:
                clip = mp.VideoFileClip(tfile.name)
                width, height = map(int, resolution.split('x'))
                resized_clip = clip.resize(newsize=(width, height))

                # Save to a temporary file
                temp_video_file = tempfile.NamedTemporaryFile(delete=False, suffix='.' + output_format)
                resized_clip.write_videofile(temp_video_file.name, codec='libx264')

                # Provide download link
                with open(temp_video_file.name, 'rb') as f:
                    st.download_button('Download Video', f, file_name='resized_video.' + output_format)
                os.unlink(temp_video_file.name)
                os.unlink(tfile.name)
            except Exception as e:
                st.error(f"An error occurred during video processing: {e}")

if __name__ == "__main__":
    main()
