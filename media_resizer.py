import streamlit as st
from PIL import Image, ImageOps
import moviepy.editor as mp
import tempfile
import os
from streamlit_cropper import st_cropper
from moviepy.video.fx.all import margin

def main():
    import sys
    st.write(f"Python version: {sys.version}")
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

    # Define platforms and use cases
    platforms = ["Instagram", "Facebook", "Youtube", "Twitter", "Snapchat", "Linkedin", "Pinterest", "Custom"]
    platform = st.selectbox("Select Platform", platforms)

    platform_use_cases = {
        "Instagram": ["Feed Square", "Feed Vertical", "Feed Full Portrait", "Stories"],
        "Facebook": ["Feed Horizontal", "Feed Square", "Feed Vertical", "Feed Full Portrait", "Cover", "Stories", "Instant Articles Horizontal", "Instant Articles Square", "Instant Articles Vertical", "Instant Articles Full Portrait"],
        "Youtube": ["Feed Horizontal", "In-Stream"],
        "Twitter": ["Feed Horizontal", "Feed Square", "Feed Vertical"],
        "Snapchat": ["Stories"],
        "Linkedin": ["Feed Horizontal", "Feed Square", "Feed Vertical"],
        "Pinterest": ["Feed Square", "Feed Vertical", "Feed Full Portrait"],
        "Custom": ["Custom"],
    }

    use_cases = platform_use_cases.get(platform, [])
    use_case = st.selectbox("Select Use Case", use_cases)

    aspect_ratios = {
        "Feed Horizontal": (16, 9),
        "Feed Square": (1, 1),
        "Feed Vertical": (4, 5),
        "Feed Full Portrait": (9, 16),
        "Cover": (16, 9),
        "In-Stream": (16, 9),
        "Stories": (9, 16),
        "Instant Articles Horizontal": (16, 9),
        "Instant Articles Square": (1, 1),
        "Instant Articles Vertical": (4, 5),
        "Instant Articles Full Portrait": (9, 16),
        "Custom": None,
    }

    if use_case == "Custom":
        # Allow user to input custom dimensions
        custom_width = st.number_input("Custom Width (pixels)", min_value=1, value=1080)
        custom_height = st.number_input("Custom Height (pixels)", min_value=1, value=1920)
        desired_width = custom_width
        desired_height = custom_height
    else:
        aspect_ratio = aspect_ratios.get(use_case, None)
        if aspect_ratio is None:
            st.error("Invalid aspect ratio selected.")
            return

        desired_width = st.number_input("Desired Width (pixels)", min_value=1, value=1080)
        desired_height = int(desired_width * aspect_ratio[1] / aspect_ratio[0])

    st.write(f"Calculated Height: {desired_height} pixels")

    # Option to choose resize method
    resize_method = st.radio("Select Resize Method", ["Crop", "Pad (Add borders)"])

    output_format = st.selectbox("Output Format", ["JPEG", "PNG", "BMP", "GIF"])

    # Resize the image while maintaining aspect ratio
    img_aspect_ratio = image.width / image.height
    target_aspect_ratio = desired_width / desired_height

    if img_aspect_ratio > target_aspect_ratio:
        # Image is wider than target aspect ratio
        scale_factor = desired_height / image.height
    else:
        # Image is taller than target aspect ratio
        scale_factor = desired_width / image.width

    new_width = int(image.width * scale_factor)
    new_height = int(image.height * scale_factor)
    resized_image = image.resize((new_width, new_height), Image.ANTIALIAS)

    if resize_method == "Crop":
        # Crop the image to the desired dimensions
        left = (new_width - desired_width) / 2
        top = (new_height - desired_height) / 2
        right = (new_width + desired_width) / 2
        bottom = (new_height + desired_height) / 2
        final_image = resized_image.crop((left, top, right, bottom))
    else:
        # Pad the image to the desired dimensions
        delta_width = desired_width - new_width
        delta_height = desired_height - new_height
        padding = (delta_width // 2, delta_height // 2, delta_width - (delta_width // 2), delta_height - (delta_height // 2))
        final_image = ImageOps.expand(resized_image, padding, fill=(0, 0, 0))

    st.image(final_image, caption='Preview of Resized Image', use_column_width=True)

    if st.button("Download Resized Image"):
        # Save to a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.' + output_format.lower())
        final_image.save(temp_file.name, output_format)

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
        "9:16": (9, 16),
        "4:5": (4, 5),
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

        # Define platforms and use cases
        platforms = ["Instagram", "Facebook", "Youtube", "Twitter", "Snapchat", "Linkedin", "Pinterest", "Custom"]
        platform = st.selectbox("Select Platform", platforms)

        platform_use_cases = {
            "Instagram": ["Feed Square", "Feed Vertical", "Feed Full Portrait", "Stories"],
            "Facebook": ["Feed Horizontal", "Feed Square", "Feed Vertical", "Feed Full Portrait", "Cover", "Stories", "Instant Articles Horizontal", "Instant Articles Square", "Instant Articles Vertical", "Instant Articles Full Portrait"],
            "Youtube": ["Feed Horizontal", "In-Stream"],
            "Twitter": ["Feed Horizontal", "Feed Square", "Feed Vertical"],
            "Snapchat": ["Stories"],
            "Linkedin": ["Feed Horizontal", "Feed Square", "Feed Vertical"],
            "Pinterest": ["Feed Square", "Feed Vertical", "Feed Full Portrait"],
            "Custom": ["Custom"],
        }

        use_cases = platform_use_cases.get(platform, [])
        use_case = st.selectbox("Select Use Case", use_cases)

        aspect_ratios = {
            "Feed Horizontal": (16, 9),
            "Feed Square": (1, 1),
            "Feed Vertical": (4, 5),
            "Feed Full Portrait": (9, 16),
            "Cover": (16, 9),
            "In-Stream": (16, 9),
            "Stories": (9, 16),
            "Instant Articles Horizontal": (16, 9),
            "Instant Articles Square": (1, 1),
            "Instant Articles Vertical": (4, 5),
            "Instant Articles Full Portrait": (9, 16),
            "Custom": None,
        }

        if use_case == "Custom":
            # Allow user to input custom dimensions
            custom_width = st.number_input("Custom Width (pixels)", min_value=1, value=1080)
            custom_height = st.number_input("Custom Height (pixels)", min_value=1, value=1920)
            desired_width = custom_width
            desired_height = custom_height
        else:
            aspect_ratio = aspect_ratios.get(use_case, None)
            if aspect_ratio is None:
                st.error("Invalid aspect ratio selected.")
                return

            desired_width = st.number_input("Desired Width (pixels)", min_value=1, value=1080)
            desired_height = int(desired_width * aspect_ratio[1] / aspect_ratio[0])

        st.write(f"Calculated Height: {desired_height} pixels")

        # Option to choose resize method
        resize_method = st.radio("Select Resize Method", ["Crop", "Pad (Add borders)"])

        output_format = st.selectbox("Output Format", ["mp4", "avi", "mov", "mkv"])

        if st.button("Resize and Convert Video"):
            try:
                clip = mp.VideoFileClip(tfile.name)

                # Resize the clip while maintaining aspect ratio
                clip_aspect_ratio = clip.w / clip.h
                target_aspect_ratio = desired_width / desired_height

                if clip_aspect_ratio > target_aspect_ratio:
                    # Video is wider than target aspect ratio
                    new_height = desired_height
                    new_width = int(clip.w * (desired_height / clip.h))
                else:
                    # Video is taller than target aspect ratio
                    new_width = desired_width
                    new_height = int(clip.h * (desired_width / clip.w))

                resized_clip = clip.resize(newsize=(new_width, new_height))

                if resize_method == "Crop":
                    # Crop the clip to the desired dimensions
                    x_center = new_width / 2
                    y_center = new_height / 2
                    x1 = x_center - desired_width / 2
                    y1 = y_center - desired_height / 2
                    x2 = x_center + desired_width / 2
                    y2 = y_center + desired_height / 2

                    cropped_clip = resized_clip.crop(x1=max(0, x1), y1=max(0, y1), x2=min(new_width, x2), y2=min(new_height, y2))
                    final_clip = cropped_clip
                else:
                    # Pad the clip to the desired dimensions
                    pad_width = desired_width - new_width
                    pad_height = desired_height - new_height
                    pad_left = int(pad_width / 2)
                    pad_right = pad_width - pad_left
                    pad_top = int(pad_height / 2)
                    pad_bottom = pad_height - pad_top

                    final_clip = margin(
                        resized_clip,
                        left=pad_left,
                        right=pad_right,
                        top=pad_top,
                        bottom=pad_bottom,
                        color=(0, 0, 0)
                    )

                # Save to a temporary file
                temp_video_file = tempfile.NamedTemporaryFile(delete=False, suffix='.' + output_format)
                temp_video_path = temp_video_file.name
                temp_video_file.close()  # Close the file so MoviePy can write to it

                # Determine the audio codec based on the output format
                if output_format == 'mp4':
                    video_codec = 'libx264'
                    audio_codec = 'aac'
                elif output_format == 'avi':
                    video_codec = 'mpeg4'
                    audio_codec = 'mp3'
                elif output_format == 'mov':
                    video_codec = 'libx264'
                    audio_codec = 'aac'
                elif output_format == 'mkv':
                    video_codec = 'libx264'
                    audio_codec = 'aac'
                else:
                    video_codec = 'libx264'
                    audio_codec = 'aac'

                # Use faster encoding preset and other optimizations
                ffmpeg_params = ['-preset', 'ultrafast', '-ac', '2']
                final_clip.write_videofile(
                    temp_video_path,
                    codec=video_codec,
                    audio_codec=audio_codec,
                    audio=True,
                    threads=4,  # Adjust based on your CPU
                    ffmpeg_params=ffmpeg_params,
                    logger=None  # Suppress verbose output
                )

                # Display the resized video
                st.write("### Resized Video Preview")
                st.video(temp_video_path)

                # Provide download link
                with open(temp_video_path, 'rb') as f:
                    st.download_button('Download Resized Video', f, file_name='resized_video.' + output_format)

                # Clean up temporary files
                os.unlink(temp_video_path)
                os.unlink(tfile.name)
            except Exception as e:
                st.error(f"An error occurred during video processing: {e}")

if __name__ == "__main__":
    main()
