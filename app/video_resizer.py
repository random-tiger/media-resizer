# app/video_resizer.py

import streamlit as st
import moviepy.editor as mp
import tempfile
import os
from moviepy.video.fx.all import margin
from utils import clean_up_files

def video_uploader():
    uploaded_video = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov", "mkv"])
    if uploaded_video is not None:
        # Save uploaded video to a temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_video.read())
        tfile.flush()
        input_video_path = tfile.name
        st.video(input_video_path)

        st.write("### Resize Options")

        # Load the video clip to get dimensions
        try:
            clip = mp.VideoFileClip(input_video_path)
            original_width = clip.w
            original_height = clip.h
            original_aspect_ratio = original_width / original_height
        except Exception as e:
            st.error(f"An error occurred while loading the video: {e}")
            clean_up_files([input_video_path])
            return

        # Define platforms and their aspect ratios
        platforms = [
            "Instagram", "Facebook", "YouTube", "Twitter",
            "Snapchat", "LinkedIn", "Pinterest", "Tubi", "Custom"
        ]
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
            # New Platform: Tubi
            "Tubi": {
                "Horizontal 16:9 (1920x1080)": (16, 9),
                "Horizontal 4K (3840x2160)": (16, 9),
                "Vertical 9:16 (1080x1920)": (9, 16),
                "Square 1:1 (1080x1080)": (1, 1),
                "Banner 3.88:1 (1628x420)": (3.88, 1),
            },
            "Custom": {},
        }

        if platform != "Custom":
            aspect_ratio_dict = platform_aspect_ratios.get(platform, {})
            aspect_ratio_names = list(aspect_ratio_dict.keys())
            selected_aspect_ratio_name = st.selectbox("Select Aspect Ratio", aspect_ratio_names)
            aspect_ratio = aspect_ratio_dict[selected_aspect_ratio_name]

            # Determine target width and height based on selected aspect ratio
            if platform == "Tubi":
                # Extract resolution from the aspect ratio name if available
                if "1920x1080" in selected_aspect_ratio_name:
                    target_width, target_height = 1920, 1080
                elif "3840x2160" in selected_aspect_ratio_name:
                    target_width, target_height = 3840, 2160
                elif "1080x1920" in selected_aspect_ratio_name:
                    target_width, target_height = 1080, 1920
                elif "1080x1080" in selected_aspect_ratio_name:
                    target_width, target_height = 1080, 1080
                elif "1628x420" in selected_aspect_ratio_name:
                    target_width, target_height = 1628, 420
                else:
                    # Default to a standard width if resolution is not specified
                    target_width = 1080
                    target_height = int(target_width / aspect_ratio[0] * aspect_ratio[1])
            else:
                # For other platforms, set a default width and calculate height
                target_width = 1080
                target_height = int(target_width / aspect_ratio[0] * aspect_ratio[1])

        else:
            # For Custom platform, set aspect ratio and determine dimensions
            common_aspect_ratios = {
                "16:9": (16, 9),
                "4:3": (4, 3),
                "1:1": (1, 1),
                "4:5": (4, 5),
                "9:16": (9, 16),
            }
            aspect_ratio_names = list(common_aspect_ratios.keys())
            selected_common_aspect_ratio = st.selectbox("Select Aspect Ratio", aspect_ratio_names)
            aspect_ratio = common_aspect_ratios[selected_common_aspect_ratio]

            # Set default width and calculate height
            target_width = 1080
            target_height = int(target_width / aspect_ratio[0] * aspect_ratio[1])

        # Display the determined dimensions to the user
        st.markdown(f"**Target Dimensions:** {target_width} x {target_height} pixels")

        # Resize method
        resize_method = st.radio("Select Resize Method", ["Crop", "Pad (Add borders)"])

        # Initialize crop position only if the resize method is Crop
        if resize_method == "Crop":
            crop_positions = [
                "Center",
                "Top",
                "Bottom",
                "Left",
                "Right",
                "Top-Left",
                "Top-Right",
                "Bottom-Left",
                "Bottom-Right"
            ]
            crop_position = st.selectbox("Select Crop Position", crop_positions)
        else:
            crop_position = None  # Not applicable

        output_format = st.selectbox("Output Format", ["mp4", "avi", "mov", "mkv"])

        if st.button("Resize and Convert Video"):
            try:
                # Use target_width and target_height
                target_width = target_width
                target_height = target_height

                # Calculate scaling factor based on resize method
                if resize_method == "Crop":
                    # For cropping, scale up to ensure dimensions are larger
                    scale_factor_w = target_width / clip.w
                    scale_factor_h = target_height / clip.h
                    scale_factor = max(scale_factor_w, scale_factor_h)
                else:
                    # For padding, scale down to ensure dimensions are smaller
                    scale_factor_w = target_width / clip.w
                    scale_factor_h = target_height / clip.h
                    scale_factor = min(scale_factor_w, scale_factor_h)

                new_width = int(clip.w * scale_factor)
                new_height = int(clip.h * scale_factor)

                resized_clip = clip.resize(newsize=(new_width, new_height))

                if resize_method == "Crop":
                    # Determine crop coordinates based on selected crop position
                    final_clip = apply_crop(resized_clip, target_width, target_height, crop_position)
                else:
                    # Pad to desired dimensions
                    pad_width = target_width - new_width
                    pad_height = target_height - new_height

                    # Ensure pad sizes are non-negative
                    pad_left = int(pad_width / 2) if pad_width > 0 else 0
                    pad_right = pad_width - pad_left if pad_width > 0 else 0
                    pad_top = int(pad_height / 2) if pad_height > 0 else 0
                    pad_bottom = pad_height - pad_top if pad_height > 0 else 0

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
                output_video_path = temp_video_file.name
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
                    output_video_path,
                    codec=video_codec,
                    audio_codec=audio_codec,
                    audio=True,
                    threads=6,  # Adjust based on your CPU
                    ffmpeg_params=ffmpeg_params,
                    logger=None  # Suppress verbose output
                )

                # Display the resized video
                st.write("### Resized Video Preview")
                st.video(output_video_path)

                # Provide download link
                with open(output_video_path, 'rb') as f:
                    st.download_button(
                        label='Download Resized Video', 
                        data=f, 
                        file_name='resized_video.' + output_format,
                        mime=f'video/{output_format}'
                    )

            except Exception as e:
                st.error(f"An error occurred during video processing: {e}")
            finally:
                # Clean up temporary files and release resources
                clip.close()
                clean_up_files([input_video_path, output_video_path])

def apply_crop(clip, target_width, target_height, position):
    """
    Apply cropping to the video clip based on the selected position.

    Args:
        clip (moviepy.editor.VideoFileClip): The resized video clip.
        target_width (int): The desired width after cropping.
        target_height (int): The desired height after cropping.
        position (str): The crop position selected by the user.

    Returns:
        moviepy.editor.VideoFileClip: The cropped video clip.
    """
    new_width, new_height = clip.size
    if new_width < target_width or new_height < target_height:
        # If the resized clip is smaller than target, pad instead of cropping
        pad_width = target_width - new_width
        pad_height = target_height - new_height

        pad_left = pad_width // 2 if pad_width > 0 else 0
        pad_right = pad_width - pad_left if pad_width > 0 else 0
        pad_top = pad_height // 2 if pad_height > 0 else 0
        pad_bottom = pad_height - pad_top if pad_height > 0 else 0

        return margin(
            clip,
            left=pad_left,
            right=pad_right,
            top=pad_top,
            bottom=pad_bottom,
            color=(0, 0, 0)
        )

    # Calculate cropping coordinates based on position
    if position == "Center":
        x1 = (new_width - target_width) / 2
        y1 = (new_height - target_height) / 2
    elif position == "Top":
        x1 = (new_width - target_width) / 2
        y1 = 0
    elif position == "Bottom":
        x1 = (new_width - target_width) / 2
        y1 = new_height - target_height
    elif position == "Left":
        x1 = 0
        y1 = (new_height - target_height) / 2
    elif position == "Right":
        x1 = new_width - target_width
        y1 = (new_height - target_height) / 2
    elif position == "Top-Left":
        x1 = 0
        y1 = 0
    elif position == "Top-Right":
        x1 = new_width - target_width
        y1 = 0
    elif position == "Bottom-Left":
        x1 = 0
        y1 = new_height - target_height
    elif position == "Bottom-Right":
        x1 = new_width - target_width
        y1 = new_height - target_height
    else:
        # Default to center if position is unrecognized
        x1 = (new_width - target_width) / 2
        y1 = (new_height - target_height) / 2

    x2 = x1 + target_width
    y2 = y1 + target_height

    # Ensure coordinates are within the frame
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(new_width, x2)
    y2 = min(new_height, y2)

    # Apply cropping
    final_clip = clip.crop(x1=x1, y1=y1, x2=x2, y2=y2)

    return final_clip
