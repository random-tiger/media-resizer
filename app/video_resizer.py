# app/video_resizer.py

import streamlit as st
import moviepy.editor as mp
import tempfile
import os
from moviepy.video.fx.all import margin
from utils import clean_up_files
from streamlit_image_coordinates import streamlit_image_coordinates
from PIL import Image

def video_uploader():
    st.title("📹 Video Resizer: Pad or Crop")

    # File uploader with size limitation
    uploaded_video = st.file_uploader(
        "Choose a video file",
        type=["mp4", "avi", "mov", "mkv"],
        accept_multiple_files=False
    )
    if uploaded_video is not None:
        # Limit file size to prevent server overload (e.g., 500MB)
        max_file_size = 500 * 1024 * 1024  # 500 MB
        if uploaded_video.size > max_file_size:
            st.error("File size exceeds the maximum limit of 500MB.")
            return

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
        resize_method = st.radio("Select Resize Method", ["Pad (Add Borders)", "Crop"])

        # Initialize crop variables
        reference_point = None  # (x, y) in original video dimensions

        if resize_method == "Crop":
            st.write("#### Select Crop Reference Point")
            st.write("Click on the video thumbnail to set the center point for cropping.")

            # Extract a thumbnail frame (e.g., at 1 second)
            try:
                thumbnail_frame = clip.get_frame(1)  # Get frame at 1 second
                thumbnail_image = Image.fromarray(thumbnail_frame)
                # Resize thumbnail for display (reduce size for better performance)
                display_scale = 0.5
                display_width = int(original_width * display_scale)
                display_height = int(original_height * display_scale)
                thumbnail_image_resized = thumbnail_image.resize((display_width, display_height))
            except Exception as e:
                st.error(f"An error occurred while extracting a thumbnail frame: {e}")
                clean_up_files([input_video_path])
                return

            # Display the thumbnail and capture click coordinates
            click_coords = streamlit_image_coordinates(thumbnail_image_resized)

            if click_coords:
                # Calculate the reference point in original video dimensions
                rel_x = click_coords['x'] / display_width
                rel_y = click_coords['y'] / display_height

                reference_x = int(rel_x * original_width)
                reference_y = int(rel_y * original_height)

                reference_point = (reference_x, reference_y)

                st.markdown(f"**Selected Reference Point:** ({reference_x}, {reference_y})")

                # Aspect Ratio or Dimensions input
                crop_option = st.radio("Specify Crop By", ["Aspect Ratio", "Exact Dimensions"])

                if crop_option == "Aspect Ratio":
                    aspect_ratio_input = st.text_input(
                        "Enter Aspect Ratio (e.g., 16:9)",
                        value="16:9"
                    )
                    try:
                        ar_w, ar_h = map(float, aspect_ratio_input.split(':'))
                        desired_aspect_ratio = ar_w / ar_h
                    except:
                        st.error("Invalid aspect ratio format. Please use W:H (e.g., 16:9).")
                        desired_aspect_ratio = None

                else:
                    exact_width = st.number_input(
                        "Enter Desired Width (pixels)",
                        min_value=1,
                        value=target_width,
                        step=1
                    )
                    exact_height = st.number_input(
                        "Enter Desired Height (pixels)",
                        min_value=1,
                        value=target_height,
                        step=1
                    )
                    desired_aspect_ratio = exact_width / exact_height if exact_height != 0 else None

                if desired_aspect_ratio:
                    st.markdown(f"**Desired Aspect Ratio:** {desired_aspect_ratio:.2f}")

        output_format = st.selectbox("Output Format", ["mp4", "avi", "mov", "mkv"])

        if st.button("Resize and Convert Video"):
            with st.spinner("Processing video..."):
                try:
                    if resize_method == "Crop":
                        if not reference_point:
                            st.error("Please click on the thumbnail to set the reference point for cropping.")
                            return

                        if crop_option == "Aspect Ratio":
                            if not desired_aspect_ratio:
                                st.error("Invalid aspect ratio. Please enter in W:H format (e.g., 16:9).")
                                return

                            # Calculate crop dimensions based on aspect ratio
                            # We can set either width or height based on the aspect ratio and ensure it's within video bounds
                            # For simplicity, let's choose the largest possible area around the reference point

                            # Determine the maximum possible width and height from the reference point
                            max_left = reference_point[0]
                            max_right = original_width - reference_point[0]
                            max_top = reference_point[1]
                            max_bottom = original_height - reference_point[1]

                            # Calculate maximum width and height based on aspect ratio
                            # Width is determined by height * aspect_ratio
                            # Height is determined by width / aspect_ratio

                            # Start with the maximum possible height
                            max_possible_height = min(max_top, max_bottom) * 2
                            calculated_width = max_possible_height * desired_aspect_ratio
                            if calculated_width > min(max_left, max_right) * 2:
                                # Adjust height based on width
                                calculated_width = min(max_left, max_right) * 2
                                calculated_height = calculated_width / desired_aspect_ratio
                            else:
                                calculated_height = max_possible_height

                            crop_width = int(calculated_width)
                            crop_height = int(calculated_height)

                            # Calculate top-left corner of the crop rectangle
                            crop_x1 = reference_point[0] - crop_width // 2
                            crop_y1 = reference_point[1] - crop_height // 2

                            # Ensure crop rectangle is within video boundaries
                            crop_x1 = max(0, crop_x1)
                            crop_y1 = max(0, crop_y1)
                            crop_x2 = min(original_width, crop_x1 + crop_width)
                            crop_y2 = min(original_height, crop_y1 + crop_height)

                            # Adjust if crop area is smaller than desired due to boundary limits
                            crop_width = crop_x2 - crop_x1
                            crop_height = crop_y2 - crop_y1

                            # Display the crop area size
                            st.markdown(f"**Final Crop Area Size:** {crop_width} x {crop_height} pixels")

                            # Perform cropping
                            final_clip = clip.crop(x1=crop_x1, y1=crop_y1, x2=crop_x2, y2=crop_y2)

                        else:
                            # Exact Dimensions
                            desired_width = int(exact_width)
                            desired_height = int(exact_height)

                            # Calculate top-left corner based on reference point
                            crop_x1 = reference_point[0] - desired_width // 2
                            crop_y1 = reference_point[1] - desired_height // 2
                            crop_x2 = crop_x1 + desired_width
                            crop_y2 = crop_y1 + desired_height

                            # Ensure crop rectangle is within video boundaries
                            if crop_x1 < 0:
                                crop_x1 = 0
                                crop_x2 = desired_width
                            if crop_y1 < 0:
                                crop_y1 = 0
                                crop_y2 = desired_height
                            if crop_x2 > original_width:
                                crop_x2 = original_width
                                crop_x1 = original_width - desired_width
                            if crop_y2 > original_height:
                                crop_y2 = original_height
                                crop_y1 = original_height - desired_height

                            # Perform cropping
                            final_clip = clip.crop(x1=crop_x1, y1=crop_y1, x2=crop_x2, y2=crop_y2)

                    else:
                        # Pad (Add Borders)
                        # Calculate scaling factor to fit the video within target dimensions
                        scale_factor_w = target_width / original_width
                        scale_factor_h = target_height / original_height
                        scale_factor = min(scale_factor_w, scale_factor_h)

                        new_width = int(original_width * scale_factor)
                        new_height = int(original_height * scale_factor)

                        # Resize the clip
                        resized_clip = clip.resize(newsize=(new_width, new_height))

                        # Calculate padding
                        pad_width = target_width - new_width
                        pad_height = target_height - new_height

                        pad_left = pad_width // 2
                        pad_right = pad_width - pad_left
                        pad_top = pad_height // 2
                        pad_bottom = pad_height - pad_top

                        # Apply padding
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

                    st.success("Video processed successfully!")

                except Exception as e:
                    st.error(f"An error occurred during video processing: {e}")
                finally:
                    # Clean up temporary files and release resources
                    clip.close()
                    # Only clean up if output_video_path is defined
                    files_to_clean = [input_video_path]
                    if 'output_video_path' in locals() and os.path.exists(output_video_path):
                        files_to_clean.append(output_video_path)
                    clean_up_files(files_to_clean)

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
