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
        dpi = st.number_input("DPI (Dots Per Inch)", min_value=1, value=int(dpi), key='dpi')
    else:
        st.session_state['dpi'] = dpi  # Ensure dpi is in session_state for consistency

    # Aspect Ratio Lock
    if 'aspect_ratio' not in st.session_state:
        st.session_state.aspect_ratio = image.width / image.height

    maintain_aspect = st.checkbox("Maintain Aspect Ratio", value=True, key='maintain_aspect')

    # Initialize width and height in session state
    if units == "Pixels":
        unit_label = "pixels"
        if 'width' not in st.session_state:
            st.session_state.width = image.width
        if 'height' not in st.session_state:
            st.session_state.height = image.height
    else:
        if units == "Inches":
            unit_conversion = st.session_state.dpi
            unit_label = "inches"
        elif units == "Centimeters":
            unit_conversion = st.session_state.dpi / 2.54
            unit_label = "centimeters"
        elif units == "Millimeters":
            unit_conversion = st.session_state.dpi / 25.4
            unit_label = "millimeters"

        # Convert pixels to selected units
        width_in_units = round(image.width / unit_conversion, 2)
        height_in_units = round(image.height / unit_conversion, 2)

        if 'width' not in st.session_state:
            st.session_state.width = width_in_units
        if 'height' not in st.session_state:
            st.session_state.height = height_in_units

    # Define callbacks for width and height inputs
    def update_width():
        if st.session_state.maintain_aspect:
            st.session_state.height = round(float(st.session_state.width) / st.session_state.aspect_ratio, 2)

    def update_height():
        if st.session_state.maintain_aspect:
            st.session_state.width = round(float(st.session_state.height) * st.session_state.aspect_ratio, 2)

    # Input width and height with callbacks
    width = st.number_input(f"Width ({unit_label})", min_value=0.01, key='width', on_change=update_width)
    height = st.number_input(f"Height ({unit_label})", min_value=0.01, key='height', on_change=update_height)

    if units == "Pixels":
        width_pixels = int(st.session_state.width)
        height_pixels = int(st.session_state.height)
    else:
        # Convert back to pixels for resizing
        width_pixels = int(float(st.session_state.width) * unit_conversion)
        height_pixels = int(float(st.session_state.height) * unit_conversion)

    output_format = st.selectbox("Output Format", ["JPEG", "PNG", "BMP", "GIF"])

    # Display the resized image as a preview
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

    if st.button("Generate Cropped Image"):
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
