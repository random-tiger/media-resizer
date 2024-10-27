import streamlit as st
import moviepy.editor as mp
import tempfile
import os
from moviepy.video.fx.all import margin
import openai
import subprocess
import platform

def main():
    st.title("Media Resizer, Cropper, and Converter")
    st.write("Upload a video file to resize, crop, convert, and add subtitles.")
    
    operation_mode = st.sidebar.selectbox("Select Operation Mode", ["Video", "Subtitle Creation Mode"])
    
    if operation_mode == "Video":
        video_uploader()
    elif operation_mode == "Subtitle Creation Mode":
        subtitle_creation_mode()

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
        
        # Initialize width and height based on default dimensions
        default_width = 1080  # Set a standard default width
        default_height = int(default_width / aspect_ratio_value)

        # Initialize or retrieve vid_width and vid_height
        if 'vid_width' not in st.session_state:
            st.session_state['vid_width'] = default_width
            st.session_state['vid_height'] = default_height
            st.session_state['last_vid_aspect_ratio_name'] = selected_aspect_ratio_name
        else:
            if st.session_state['last_vid_aspect_ratio_name'] != selected_aspect_ratio_name:
                st.session_state['vid_width'] = default_width
                st.session_state['vid_height'] = default_height
                st.session_state['last_vid_aspect_ratio_name'] = selected_aspect_ratio_name
        
        vid_width = st.session_state['vid_width']
        vid_height = st.session_state['vid_height']
        
        # Input fields with unique keys
        col1, col2 = st.columns(2)
        with col1:
            vid_width = st.number_input("Width (pixels)", min_value=1, value=vid_width, key='vid_width_input')
        with col2:
            if link_aspect:
                vid_height = int(vid_width / aspect_ratio_value)
                st.markdown(f"**Height (pixels): {vid_height}**")
            else:
                vid_height = st.number_input("Height (pixels)", min_value=1, value=vid_height, key='vid_height_input')
        
        # Update session_state after inputs
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

                # Calculate scaling factor differently based on resize method
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

                    # Ensure pad sizes are non-negative
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
                    threads=12,  # Adjust based on your CPU
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
                if os.path.exists(temp_video_path):
                    os.unlink(temp_video_path)
                if os.path.exists(tfile.name):
                    os.unlink(tfile.name)
    else:
        st.write("Please upload a video file.")

def subtitle_creation_mode():
    st.header("Subtitle Creation Mode")
    
    # Check for OpenAI API key in st.secrets
    if "OPENAI_API_KEY" in st.secrets:
        openai.api_key = st.secrets["OPENAI_API_KEY"]
    else:
        st.error("OpenAI API key not found in st.secrets. Please add it to your Streamlit secrets.")
        return
    
    # Prompt user to upload a video file
    uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])
    if uploaded_video is not None:
        # Save the uploaded video to a temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_video.read())
        tfile.flush()
        
        # Display the video
        st.video(tfile.name)
        
        # Extract audio from video
        st.write("Extracting audio from the video...")
        try:
            audio_file_path = extract_audio(tfile.name)
            st.write("Audio extracted successfully.")
        except Exception as e:
            st.error(f"Error extracting audio: {e}")
            return
        
        # Generate subtitles
        st.write("Generating subtitles using OpenAI Whisper...")
        # Option to select languages
        languages = ["English", "Spanish"]
        selected_languages = st.multiselect("Select languages for subtitles", languages, default=languages)
        
        # Map language names to OpenAI language codes
        language_codes = {"English": "en", "Spanish": "es"}
        
        if not selected_languages:
            st.warning("Please select at least one language for subtitles.")
            return
        
        # Create a placeholder to show progress
        progress_bar = st.progress(0)
        total_languages = len(selected_languages)
        current_progress = 0
        
        # Dictionary to store subtitles
        subtitles = {}
        
        for language in selected_languages:
            code = language_codes[language]
            st.write(f"Transcribing audio to {language}...")
            try:
                subtitle_content = generate_subtitles(audio_file_path, code)
                st.write(f"Subtitles in {language} generated successfully.")
                subtitles[language] = subtitle_content
            except Exception as e:
                st.error(f"Error generating subtitles in {language}: {e}")
            current_progress += 1
            progress_bar.progress(current_progress / total_languages)
        
        # Provide download buttons for subtitles
        st.write("### Download Subtitles")
        for language in subtitles:
            subtitle_content = subtitles[language]
            srt_filename = f"subtitles_{language}.srt"
            st.download_button(f"Download {language} Subtitles (.srt)", subtitle_content, file_name=srt_filename)
        
        # Option to embed subtitles into the video
        st.write("### Embed Subtitles into Video")
        embed_option = st.radio("Do you want to embed subtitles into the video?", ["No", "Yes"])
        
        if embed_option == "Yes":
            language_to_embed = st.selectbox("Select language to embed into video", selected_languages)
            st.write(f"Embedding {language_to_embed} subtitles into the video...")
            try:
                subtitled_video_path = embed_subtitles_into_video(tfile.name, subtitles[language_to_embed], language_to_embed)
                st.video(subtitled_video_path)
                # Provide download button
                with open(subtitled_video_path, 'rb') as f:
                    st.download_button(f"Download Video with {language_to_embed} Subtitles", f, file_name=f"video_with_{language_to_embed}_subtitles.mp4")
            except Exception as e:
                st.error(f"Error embedding subtitles: {e}")
        else:
            st.write("Subtitles not embedded.")
        
        # Clean up temporary files
        if os.path.exists(audio_file_path):
            os.unlink(audio_file_path)
        if os.path.exists(tfile.name):
            os.unlink(tfile.name)
    else:
        st.write("Please upload a video file.")

def extract_audio(video_file_path):
    # Use moviepy to extract audio
    clip = mp.VideoFileClip(video_file_path)
    audio = clip.audio
    audio_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
    audio.write_audiofile(audio_file.name)
    clip.close()
    audio.close()
    return audio_file.name

def generate_subtitles(audio_file_path, language_code):
    with open(audio_file_path, 'rb') as audio_file:
        response = openai.Audio.transcribe(
            model="whisper-1",
            file=audio_file,
            response_format="srt",
            language=language_code
        )
    return response  # The response contains the SRT content as text

def embed_subtitles_into_video(video_file_path, subtitle_content, language):
    # Save the subtitles to a .srt file
    subtitle_file = tempfile.NamedTemporaryFile(delete=False, suffix='.srt', mode='w', encoding='utf-8')
    subtitle_file.write(subtitle_content)
    subtitle_file.close()
    
    # Output video file path
    output_video_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    
    # Handle paths differently based on OS
    if platform.system() == 'Windows':
        # For Windows, need to escape backslashes and colons
        subtitle_path = subtitle_file.name.replace('\\', '\\\\').replace(':', '\\:')
        vf_arg = f"subtitles={subtitle_path}"
    else:
        vf_arg = f"subtitles='{subtitle_file.name}'"
    
    # Use ffmpeg to embed subtitles
    cmd = [
        'ffmpeg',
        '-i', video_file_path,
        '-vf', vf_arg,
        '-c:a', 'copy',
        output_video_file.name,
        '-y'  # Overwrite output file if it exists
    ]
    
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    if result.returncode != 0:
        error_message = result.stderr.decode()
        raise Exception(f"ffmpeg error: {error_message}")
    
    # Return the path to the output video file
    return output_video_file.name

if __name__ == "__main__":
    main()
