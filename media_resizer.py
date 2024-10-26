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

    # Aspect Ratio Lock
    if 'aspect_ratio' not in st.session_state:
        st.session_state.aspect_ratio = image.width / image.height

    maintain_aspect = st.checkbox("Maintain Aspect Ratio", value=True)

    # Define callbacks for width and height inputs
    def update_width():
        if maintain_aspect:
            if units == "Pixels":
                st.session_state['height'] = int(st.session_state['width'] / st.session_state.aspect_ratio)
            else:
                st.session_state['height'] = round(st.session_state['width'] / st.session_state.aspect_ratio, 2)

    def update_height():
        if maintain_aspect:
            if units == "Pixels":
                st.session_state['width'] = int(st.session_state['height'] * st.session_state.aspect_ratio)
            else:
                st.session_state['width'] = round(st.session_state['height'] * st.session_state.aspect_ratio, 2)

    if units == "Pixels":
        # Input width and height in pixels with callbacks
        width = st.number_input("Width (pixels)", min_value=1, value=image.width, key='width', on_change=update_width)
        height = st.number_input("Height (pixels)", min_value=1, value=image.height, key='height', on_change=update_height)
        width_pixels = int(width)
        height_pixels = int(height)
    else:
        # Convert pixels to selected units
        if units == "Inches":
            unit_conversion = dpi
        elif units == "Centimeters":
            unit_conversion = dpi / 2.54
        elif units == "Millimeters":
            unit_conversion = dpi / 25.4

        width_in_units = round(image.width / unit_conversion, 2)
        height_in_units = round(image.height / unit_conversion, 2)

        # Input width and height in selected units with callbacks
        width = st.number_input(f"Width ({units.lower()})", min_value=0.01, value=width_in_units, key='width', on_change=update_width)
        height = st.number_input(f"Height ({units.lower()})", min_value=0.01, value=height_in_units, key='height', on_change=update_height)

        # Convert back to pixels for resizing
        width_pixels = int(width * unit_conversion)
        height_pixels = int(height * unit_conversion)

    output_format = st.selectbox("Output Format", ["JPEG", "PNG", "BMP", "GIF"])

    # Display the resized image as a preview
    if 'resized_image' not in st.session_state:
        st.session_state['resized_image'] = image

    # Perform resizing and display preview
    resized_image = image.resize((width_pixels, height_pixels))
    st.image(resized_image, caption='Preview of Resized Image', use_column_width=True)

    if st.button("Download Resized Image"):
        # Save to a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.' + output_format.lower())
        resized_image.save(temp_file.name, output_format)

        # Provide download link
        with open(temp_file.name, 'rb') as f:
            st.download_button('Download Image', f, file_name='resized_image.' + output_format.lower())
        os.unlink(temp_file.name)

def crop_image(image):
    st.write("### Crop Options")

    # Aspect Ratio Selection
    aspect_ratios = {
        "Free": None,
        "1:1 (Square)": (1, 1),
        "16:9": (16, 9),
        "4:3": (4, 3),
        "Custom": "Custom"
    }

    aspect_ratio_option = st.selectbox("Select Aspect Ratio", list(aspect_ratios.keys()))
    aspect_ratio = aspect_ratios[aspect_ratio_option]

    if aspect_ratio == "Custom":
        custom_width = st.number_input("Aspect Ratio Width", min_value=1, value=1)
        custom_height = st.number_input("Aspect Ratio Height", min_value=1, value=1)
        aspect_ratio = (custom_width, custom_height)

    # Get cropping coordinates using st_cropper
    realtime_update = st.checkbox("Update in Real Time", value=True)
    box_color = st.color_picker("Box Color", "#0000FF")
    aspect_ratio_value = None if aspect_ratio is None else aspect_ratio[0] / aspect_ratio[1]

    # Perform cropping
    cropped_image = st_cropper(image, realtime_update=realtime_update, box_color=box_color, aspect_ratio=aspect_ratio_value)
    st.write("Preview")
    st.image(cropped_image, use_column_width=True)

    output_format = st.selectbox("Output Format", ["JPEG", "PNG", "BMP", "GIF"])

    if st.button("Download Cropped Image"):
        # Save to a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.' + output_format.lower())
        cropped_image.save(temp_file.name, output_format)

        # Provide download link
        with open(temp_file.name, 'rb') as f:
            st.download_button('Download Image', f, file_name='cropped_image.' + output_format.lower())
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
