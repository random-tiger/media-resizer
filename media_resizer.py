import streamlit as st
from PIL import Image, ImageOps
import moviepy.editor as mp
import tempfile
import os
from streamlit_cropper import st_cropper
from moviepy.video.fx.all import margin

def main():
    st.title("Media Resizer, Cropper, and Converter")
    st.write("Upload an image or video file to resize, crop, and convert it to different formats.")

    media_type = st.sidebar.selectbox("Select Media Type", ["Video", "Image"])

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

    # Define platforms and their aspect ratios
    platforms = ["Instagram", "Facebook", "YouTube", "Twitter", "Snapchat", "LinkedIn", "Pinterest", "Custom"]
    platform = st.selectbox("Select Platform", platforms)

    # Aspect ratios for each platform
    platform_aspect_ratios = {
        "Instagram": {
            "Feed Landscape (16:9)": (16, 9),
            "Feed Square (1:1)": (1, 1),
            "Feed Portrait (4:5)": (4, 5),
            "Stories (9:16)": (9, 16),
            "IGTV (9:16)": (9, 16),
            "Ads Landscape (16:9)": (16, 9),
            "Ads Square (1:1)": (1, 1),
            "Ads Portrait (4:5)": (4, 5),
        },
        "Facebook": {
            "Feed Landscape (16:9)": (16, 9),
            "Feed Square (1:1)": (1, 1),
            "Feed Portrait (4:5)": (4, 5),
            "Stories (9:16)": (9, 16),
            "Cover (16:9)": (16, 9),
            "Ads Landscape (16:9)": (16, 9),
            "Ads Square (1:1)": (1, 1),
            "Ads Portrait (4:5)": (4, 5),
        },
        "YouTube": {
            "Standard (16:9)": (16, 9),
        },
        "Twitter": {
            "Feed Landscape (16:9)": (16, 9),
            "Feed Square (1:1)": (1, 1),
            "Feed Portrait (4:5)": (4, 5),
        },
        "Snapchat": {
            "Stories (9:16)": (9, 16),
        },
        "LinkedIn": {
            "Feed Landscape (16:9)": (16, 9),
            "Feed Square (1:1)": (1, 1),
            "Feed Portrait (4:5)": (4, 5),
        },
        "Pinterest": {
            "Standard Pin (2:3)": (2, 3),
            "Square Pin (1:1)": (1, 1),
            "Long Pin (1:2.1)": (1, 2.1),
        },
        "Custom": {},
    }

    if platform != "Custom":
        aspect_ratio_dict = platform_aspect_ratios.get(platform, {})
        aspect_ratio_names = list(aspect_ratio_dict.keys())
        selected_aspect_ratio_name = st.selectbox("Select Aspect Ratio", aspect_ratio_names)
        aspect_ratio = aspect_ratio_dict[selected_aspect_ratio_name]
    else:
        # For Custom platform
        common_aspect_ratios = {
            "16:9": (16, 9),
            "4:3": (4, 3),
            "1:1": (1, 1),
            "4:5": (4, 5),
            "9:16": (9, 16),
            "Custom": "Custom"
        }
        aspect_ratio_names = list(common_aspect_ratios.keys())
        selected_common_aspect_ratio = st.selectbox("Select Aspect Ratio", aspect_ratio_names)
        if selected_common_aspect_ratio != "Custom":
            aspect_ratio = common_aspect_ratios[selected_common_aspect_ratio]
        else:
            # User inputs custom aspect ratio
            custom_width = st.number_input("Aspect Ratio Width", min_value=1, value=16)
            custom_height = st.number_input("Aspect Ratio Height", min_value=1, value=9)
            aspect_ratio = (custom_width, custom_height)

    # Aspect ratio value
    aspect_ratio_value = aspect_ratio[0] / aspect_ratio[1]

    # Link or unlink aspect ratio
    link_aspect = st.checkbox("Link Aspect Ratio", value=True)

    # Initialize width and height
    default_width = 1080
    default_height = int(default_width / aspect_ratio_value)

    # Reset width and height when aspect ratio changes
    if 'last_aspect_ratio_name' not in st.session_state:
        st.session_state.last_aspect_ratio_name = selected_aspect_ratio_name
        width = default_width
        height = default_height
    else:
        if st.session_state.last_aspect_ratio_name != selected_aspect_ratio_name:
            st.session_state.last_aspect_ratio_name = selected_aspect_ratio_name
            width = default_width
            height = default_height
        else:
            # Use previous width and height from session_state, or defaults
            width = st.session_state.get('width', default_width)
            height = st.session_state.get('height', default_height)

    # Input fields
    col1, col2 = st.columns(2)
    with col1:
        width = st.number_input("Width (pixels)", min_value=1, value=width)
    with col2:
        if link_aspect:
            height = int(width / aspect_ratio_value)
            st.markdown(f"**Height (pixels): {height}**")
        else:
            height = st.number_input("Height (pixels)", min_value=1, value=height)

    # Update session_state
    st.session_state['width'] = width
    st.session_state['height'] = height

    # Resize method
    resize_method = st.radio("Select Resize Method", ["Crop", "Pad (Add borders)"])

    output_format = st.selectbox("Output Format", ["JPEG", "PNG", "BMP", "GIF"])

    # Resize image
    img_aspect_ratio = image.width / image.height
    target_aspect_ratio = width / height

    if img_aspect_ratio > target_aspect_ratio:
        # Image is wider
        scale_factor = height / image.height
    else:
        # Image is taller
        scale_factor = width / image.width

    new_width = int(image.width * scale_factor)
    new_height = int(image.height * scale_factor)
    resized_image = image.resize((new_width, new_height), Image.ANTIALIAS)

    if resize_method == "Crop":
        # Crop to desired dimensions
        left = (new_width - width) / 2
        top = (new_height - height) / 2
        right = (new_width + width) / 2
        bottom = (new_height + height) / 2
        final_image = resized_image.crop((left, top, right, bottom))
    else:
        # Pad to desired dimensions
        delta_width = width - new_width
        delta_height = height - new_height
        padding = (
            delta_width // 2,
            delta_height // 2,
            delta_width - (delta_width // 2),
            delta_height - (delta_height // 2)
        )
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
        custom_width = st.number_input("Aspect Ratio Width", min_value=1, value=16)
        custom_height = st.number_input("Aspect Ratio Height", min_value=1, value=9)
        aspect_ratio_value = custom_width / custom_height
    elif aspect_ratio is not None:
        aspect_ratio_value = aspect_ratio[0] / aspect_ratio[1]
    else:
        aspect_ratio_value = None

    # Get cropping coordinates using st_cropper
    realtime_update = st.checkbox("Update in Real Time", value=True)
    box_color = st.color_picker("Box Color", "#0000FF")

    # Perform cropping
    cropped_image = st_cropper(
        image,
        realtime_update=realtime_update,
        box_color=box_color,
        aspect_ratio=aspect_ratio_value
    )
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

        # Load the video clip to get dimensions
        try:
            clip = mp.VideoFileClip(tfile.name)
            original_width = clip.w
            original_height = clip.h
            original_aspect_ratio = original_width / original_height
        except Exception as e:
            st.error(f"An error occurred while loading the video: {e}")
            return

        # Define platforms and their aspect ratios
        platforms = ["Instagram", "Facebook", "YouTube", "Twitter", "Snapchat", "LinkedIn", "Pinterest", "Custom"]
        platform = st.selectbox("Select Platform", platforms)

        platform_aspect_ratios = {
            "Instagram": {
                "Feed Landscape (16:9)": (16, 9),
                "Feed Square (1:1)": (1, 1),
                "Feed Portrait (4:5)": (4, 5),
                "Stories (9:16)": (9, 16),
                "IGTV (9:16)": (9, 16),
                "Ads Landscape (16:9)": (16, 9),
                "Ads Square (1:1)": (1, 1),
                "Ads Portrait (4:5)": (4, 5),
            },
            "Facebook": {
                "Feed Landscape (16:9)": (16, 9),
                "Feed Square (1:1)": (1, 1),
                "Feed Portrait (4:5)": (4, 5),
                "Stories (9:16)": (9, 16),
                "Cover (16:9)": (16, 9),
                "Ads Landscape (16:9)": (16, 9),
                "Ads Square (1:1)": (1, 1),
                "Ads Portrait (4:5)": (4, 5),
            },
            "YouTube": {
                "Standard (16:9)": (16, 9),
            },
            "Twitter": {
                "Feed Landscape (16:9)": (16, 9),
                "Feed Square (1:1)": (1, 1),
                "Feed Portrait (4:5)": (4, 5),
            },
            "Snapchat": {
                "Stories (9:16)": (9, 16),
            },
            "LinkedIn": {
                "Feed Landscape (16:9)": (16, 9),
                "Feed Square (1:1)": (1, 1),
                "Feed Portrait (4:5)": (4, 5),
            },
            "Pinterest": {
                "Standard Pin (2:3)": (2, 3),
                "Square Pin (1:1)": (1, 1),
                "Long Pin (1:2.1)": (1, 2.1),
            },
            "Custom": {},
        }

        if platform != "Custom":
            aspect_ratio_dict = platform_aspect_ratios.get(platform, {})
            aspect_ratio_names = list(aspect_ratio_dict.keys())
            selected_aspect_ratio_name = st.selectbox("Select Aspect Ratio", aspect_ratio_names)
            aspect_ratio = aspect_ratio_dict[selected_aspect_ratio_name]
        else:
            # For Custom platform
            common_aspect_ratios = {
                "16:9": (16, 9),
                "4:3": (4, 3),
                "1:1": (1, 1),
                "4:5": (4, 5),
                "9:16": (9, 16),
                "Custom": "Custom"
            }
            aspect_ratio_names = list(common_aspect_ratios.keys())
            selected_common_aspect_ratio = st.selectbox("Select Aspect Ratio", aspect_ratio_names)
            if selected_common_aspect_ratio != "Custom":
                aspect_ratio = common_aspect_ratios[selected_common_aspect_ratio]
            else:
                # User inputs custom aspect ratio
                custom_width = st.number_input("Aspect Ratio Width", min_value=1, value=16)
                custom_height = st.number_input("Aspect Ratio Height", min_value=1, value=9)
                aspect_ratio = (custom_width, custom_height)

        # Aspect ratio value
        aspect_ratio_value = aspect_ratio[0] / aspect_ratio[1]

        # Link or unlink aspect ratio
        link_aspect = st.checkbox("Link Aspect Ratio", value=True)

        # Initialize width and height based on original video dimensions
        if 'last_vid_aspect_ratio_name' not in st.session_state:
            st.session_state.last_vid_aspect_ratio_name = selected_aspect_ratio_name
            vid_width = int(original_width)
            vid_height = int(original_height)
        else:
            if st.session_state.last_vid_aspect_ratio_name != selected_aspect_ratio_name:
                st.session_state.last_vid_aspect_ratio_name = selected_aspect_ratio_name
                vid_width = int(original_width)
                vid_height = int(vid_width / aspect_ratio_value)
            else:
                vid_width = st.session_state.get('vid_width', int(original_width))
                vid_height = st.session_state.get('vid_height', int(original_height))

        # Input fields
        col1, col2 = st.columns(2)
        with col1:
            vid_width = st.number_input("Width (pixels)", min_value=1, value=vid_width)
        with col2:
            if link_aspect:
                vid_height = int(vid_width / aspect_ratio_value)
                st.markdown(f"**Height (pixels): {vid_height}**")
            else:
                vid_height = st.number_input("Height (pixels)", min_value=1, value=vid_height)

        # Update session_state
        st.session_state['vid_width'] = vid_width
        st.session_state['vid_height'] = vid_height

        # Resize method
        resize_method = st.radio("Select Resize Method", ["Crop", "Pad (Add borders)"])

        output_format = st.selectbox("Output Format", ["mp4", "avi", "mov", "mkv"])

        if st.button("Resize and Convert Video"):
            try:
                # Use vid_width and vid_height
                target_width = vid_width
                target_height = vid_height

                # Calculate scaling factor
                scale_factor_w = target_width / clip.w
                scale_factor_h = target_height / clip.h

                scale_factor = max(scale_factor_w, scale_factor_h)

                new_width = int(clip.w * scale_factor)
                new_height = int(clip.h * scale_factor)

                resized_clip = clip.resize(newsize=(new_width, new_height))

                if resize_method == "Crop":
                    # Crop to desired dimensions
                    x_center = new_width / 2
                    y_center = new_height / 2
                    x1 = x_center - target_width / 2
                    y1 = y_center - target_height / 2
                    x2 = x_center + target_width / 2
                    y2 = y_center + target_height / 2

                    final_clip = resized_clip.crop(
                        x1=max(0, x1),
                        y1=max(0, y1),
                        x2=min(new_width, x2),
                        y2=min(new_height, y2)
                    )
                else:
                    # Pad to desired dimensions
                    pad_width = target_width - new_width
                    pad_height = target_height - new_height
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

            except Exception as e:
                st.error(f"An error occurred during video processing: {e}")
            finally:
                # Clean up temporary files and release resources
                clip.close()
                os.unlink(temp_video_path)
                os.unlink(tfile.name)
    else:
        st.write("Please upload a video file.")

if __name__ == "__main__":
    main()
