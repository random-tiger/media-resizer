import streamlit as st
from PIL import Image
import moviepy.editor as mp
import tempfile
import os

def main():
    st.title("Media Resizer and Converter")
    st.write("Upload an image or video file to resize and convert it to different formats.")

    media_type = st.sidebar.selectbox("Select Media Type", ["Image", "Video"])

    if media_type == "Image":
        image_uploader()
    else:
        video_uploader()

def image_uploader():
    uploaded_image = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png", "bmp", "gif"])
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption='Original Image', use_column_width=True)

        st.write("### Resize Options")

        # Select unit of measurement
        units = st.selectbox("Units", ["Pixels", "Inches", "Centimeters", "Millimeters"])

        # Get image DPI
        if 'dpi' in image.info:
            dpi = image.info['dpi'][0]
        else:
            dpi = 72  # Default DPI

        if units != "Pixels":
            # Input DPI if units are not pixels
            dpi = st.number_input("DPI (Dots Per Inch)", min_value=1, value=int(dpi))

        # Input width and height based on selected units
        if units == "Pixels":
            width = st.number_input("Width (pixels)", min_value=1, value=image.width)
            height = st.number_input("Height (pixels)", min_value=1, value=image.height)
        else:
            width_input = st.number_input(f"Width ({units.lower()})", min_value=0.01, value=round(image.width / dpi, 2))
            height_input = st.number_input(f"Height ({units.lower()})", min_value=0.01, value=round(image.height / dpi, 2))

            # Convert units to pixels
            if units == "Inches":
                width = int(width_input * dpi)
                height = int(height_input * dpi)
            elif units == "Centimeters":
                width = int((width_input / 2.54) * dpi)
                height = int((height_input / 2.54) * dpi)
            elif units == "Millimeters":
                width = int((width_input / 25.4) * dpi)
                height = int((height_input / 25.4) * dpi)

        output_format = st.selectbox("Output Format", ["JPEG", "PNG", "BMP", "GIF"])

        if st.button("Resize and Convert Image"):
            resized_image = image.resize((int(width), int(height)))
            st.image(resized_image, caption='Resized Image', use_column_width=True)

            # Save to a temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.' + output_format.lower())
            resized_image.save(temp_file.name, output_format)

            # Provide download link
            with open(temp_file.name, 'rb') as f:
                st.download_button('Download Image', f, file_name='resized_image.' + output_format.lower())
            os.unlink(temp_file.name)

def video_uploader():
    uploaded_video = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov", "mkv"])
    if uploaded_video is not None:
        # Save uploaded video to a temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        st.video(uploaded_video)

        st.write("### Resize Options")
        resolution = st.selectbox("Resolution", ["1920x1080", "1280x720", "854x480", "640x360"])
        output_format = st.selectbox("Output Format", ["mp4", "avi", "mov", "mkv"])

        if st.button("Resize and Convert Video"):
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

if __name__ == "__main__":
    main()
