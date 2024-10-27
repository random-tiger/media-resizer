import streamlit as st
import moviepy.editor as mp
import tempfile
import os
import subprocess
import platform
import boto3
import cv2
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector
from moviepy.video.fx.all import margin
from PIL import Image
from openai import OpenAI
import requests


def main():
    st.title("Media Resizer, Converter, and Scene Search")
    st.write("Upload a video file to resize, convert, add subtitles, or search scenes based on prompts.")
    
    operation_mode = st.sidebar.selectbox(
        "Select Operation Mode", 
        ["Video", "Subtitle Creation Mode", "Scene Search"]
    )
    
    if operation_mode == "Video":
        video_uploader()
    elif operation_mode == "Subtitle Creation Mode":
        subtitle_creation_mode()
    elif operation_mode == "Scene Search":
        scene_search_mode()
        
def is_url_accessible(url):
    try:
        response = requests.head(url, timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        return False

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
            st.session_state['last_vid_aspect_ratio_name'] = selected_aspect_ratio_name if platform != "Custom" else "Custom"
        else:
            if platform != "Custom":
                if st.session_state['last_vid_aspect_ratio_name'] != selected_aspect_ratio_name:
                    st.session_state['vid_width'] = default_width
                    st.session_state['vid_height'] = default_height
                    st.session_state['last_vid_aspect_ratio_name'] = selected_aspect_ratio_name
            else:
                if st.session_state['last_vid_aspect_ratio_name'] != "Custom":
                    st.session_state['vid_width'] = default_width
                    st.session_state['vid_height'] = default_height
                    st.session_state['last_vid_aspect_ratio_name'] = "Custom"
        
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
                    threads=6,  # Adjust based on your CPU
                    ffmpeg_params=ffmpeg_params,
                    logger=None  # Suppress verbose output
                )

                # Display the resized video
                st.write("### Resized Video Preview")
                st.video(temp_video_path)

                # Provide download link
                with open(temp_video_path, 'rb') as f:
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
        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
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
                subtitle_content = generate_subtitles(audio_file_path, code, client)
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
            st.download_button(
                label=f"Download {language} Subtitles (.srt)", 
                data=subtitle_content, 
                file_name=srt_filename,
                mime="text/plain"
            )
        
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
                    st.download_button(
                        label=f"Download Video with {language_to_embed} Subtitles", 
                        data=f, 
                        file_name=f"video_with_{language_to_embed}_subtitles.mp4",
                        mime="video/mp4"
                    )
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

def generate_subtitles(audio_file_path, language_code, client):
    with open(audio_file_path, 'rb') as audio_file:
        try:
            response = client.audio.transcriptions.create(
                file=audio_file,
                model="whisper-1",
                response_format="srt",
                language=language_code
            )
            return response['text']  # The response contains the SRT content as text
        except Exception as e:
            st.error(f"Error during transcription: {e}")
            return None

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

def scene_search_mode():
    st.header("Scene Search Mode")
    
    # Check for AWS credentials and OpenAI API key in st.secrets
    if ("AWS_ACCESS_KEY_ID" in st.secrets and 
        "AWS_SECRET_ACCESS_KEY" in st.secrets and
        "AWS_S3_BUCKET_NAME" in st.secrets and
        "OPENAI_API_KEY" in st.secrets):
        
        aws_access_key_id = st.secrets["AWS_ACCESS_KEY_ID"]
        aws_secret_access_key = st.secrets["AWS_SECRET_ACCESS_KEY"]
        s3_bucket_name = st.secrets["AWS_S3_BUCKET_NAME"]
        s3_region = st.secrets.get("AWS_REGION", "us-east-1")
        openai_api_key = st.secrets["OPENAI_API_KEY"]
    else:
        st.error("AWS credentials or OpenAI API key not found in st.secrets. Please add them to your Streamlit secrets.")
        return
    
    # Initialize S3 client
    s3 = boto3.client(
        's3',
        region_name=s3_region,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
    )
    
    # Initialize OpenAI client
    client = OpenAI(api_key=openai_api_key)
    
    # Prompt user to upload a video file
    uploaded_video = st.file_uploader("Upload a video file for scene search", type=["mp4", "avi", "mov", "mkv"])
    if uploaded_video is not None:
        # Save the uploaded video to a temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_video.read())
        tfile.flush()
        
        # Display the video
        st.video(tfile.name)
        
        # Extract scenes from the video
        st.write("Extracting scenes from the video...")
        try:
            scene_list = extract_scenes(tfile.name)
            st.write(f"Extracted {len(scene_list)} scenes.")
        except Exception as e:
            st.error(f"Error extracting scenes: {e}")
            return
        
        if not scene_list:
            st.warning("No scenes were detected in the video.")
            return
        
        # Process scenes
        st.write("Processing scenes...")
        scene_data_list = []
        progress_bar = st.progress(0)
        total_scenes = len(scene_list)
        
        # Process scenes sequentially to manage resource usage
        for idx, scene_filename in enumerate(scene_list):
            data = process_scene(scene_filename, s3, s3_bucket_name, s3_region, client)
            if data:
                scene_data_list.append(data)
            progress_bar.progress((idx + 1) / total_scenes)
        
        if not scene_data_list:
            st.error("No scenes were processed successfully.")
            return
        
        # Create DataFrame
        df_scenes = pd.DataFrame(scene_data_list)
        
        # User provides a prompt
        st.write("### Search Scenes")
        prompt = st.text_input("Enter a prompt to search for scenes:")
        if prompt:
            # Search for relevant scenes
            results = search_scenes(prompt, df_scenes, client)
            
            if results.empty:
                st.write("No matching scenes found.")
            else:
                # Display the results
                st.write("### Search Results")
                for index, row in results.iterrows():
                    st.write(f"**Caption:** {row['caption']}")
                    st.write(f"**Similarity Score:** {row['similarity']:.4f}")
                    st.video(row['scene_url'])
        else:
            st.write("Please enter a prompt to search for scenes.")
        
        # Clean up temporary files
        if os.path.exists(tfile.name):
            os.unlink(tfile.name)
    else:
        st.write("Please upload a video file.")

def extract_scenes(video_path):
    # Use PySceneDetect to detect scenes
    scene_list = []
    try:
        video = open_video(video_path)
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=30.0))  # Adjust threshold as needed
        scene_manager.detect_scenes(video)
        scene_list_data = scene_manager.get_scene_list()
    except Exception as e:
        st.error(f"Error during scene detection: {e}")
        return []
    
    if not scene_list_data:
        st.warning("No scenes were detected.")
        return []
    else:
        # Save each scene as a separate video file
        scene_filenames = []
        for i, scene in enumerate(scene_list_data):
            start_time = scene[0].get_seconds()
            end_time = scene[1].get_seconds()
            if end_time - start_time < 0.5:
                continue  # Skip very short scenes
            output_filename = f"scene_{i}.mp4"
            output_filepath = os.path.join(tempfile.gettempdir(), output_filename)
            cmd = [
                'ffmpeg',
                '-y',
                '-i', video_path,
                '-ss', str(start_time),
                '-to', str(end_time),
                '-c:v', 'libx264',
                '-c:a', 'aac',
                output_filepath
            ]
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            scene_filenames.append(output_filepath)
        return scene_filenames

def process_scene(scene_filename, s3, s3_bucket_name, s3_region, client):
    # Upload scene clip to S3 and get the URL
    scene_url = upload_file_to_s3(scene_filename, s3, s3_bucket_name, s3_region, folder='scenes')
    
    # Extract a frame from the scene
    frame_filename = extract_frame_from_scene(scene_filename)
    if frame_filename is None:
        return None

    # Upload the frame image to S3 and get the pre-signed URL
    frame_url = upload_file_to_s3(frame_filename, s3, s3_bucket_name, s3_region, folder='frames')
    
    # Verify if the image URL is accessible
    if not is_url_accessible(frame_url):
        st.error(f"Generated image URL is not accessible: {frame_url}")
        return None
    
    # Generate caption using GPT-4o with the frame URL
    caption = generate_caption(frame_url, client)
    if caption is None:
        return None

    # Generate embedding using ADA embeddings model
    embedding = get_embedding(caption, client)
    if embedding is None:
        return None

    # Return scene data
    return {
        'scene_filename': scene_filename,
        'scene_url': scene_url,
        'caption': caption,
        'embedding': embedding
    }

def extract_frame_from_scene(scene_filename):
    cap = cv2.VideoCapture(scene_filename)
    if not cap.isOpened():
        st.error(f"Failed to open video file {scene_filename}")
        return None
    ret, frame = cap.read()
    cap.release()
    if ret:
        frame_filename = scene_filename + "_frame.jpg"
        
        # Optional: Resize the frame to reduce size and prevent timeout
        desired_width = 800  # Adjust as needed
        height, width, channels = frame.shape
        aspect_ratio = width / height
        desired_height = int(desired_width / aspect_ratio)
        resized_frame = cv2.resize(frame, (desired_width, desired_height))
        
        cv2.imwrite(frame_filename, resized_frame)
        
        # Verify the saved image is not corrupted
        try:
            with Image.open(frame_filename) as img:
                img.verify()  # Will raise an exception if the image is corrupted
        except Exception as e:
            st.error(f"Extracted frame is corrupted: {e}")
            os.remove(frame_filename)
            return None
        
        return frame_filename
    else:
        st.error(f"Failed to read frame from {scene_filename}")
        return None

def upload_file_to_s3(file_path, s3, s3_bucket, s3_region, folder='files', expiration=3600):
    file_name = os.path.basename(file_path)
    s3_key = f"{folder}/{file_name}"
    # Determine content type based on file extension
    _, ext = os.path.splitext(file_name)
    content_type = 'application/octet-stream'
    if ext.lower() in ['.jpg', '.jpeg']:
        content_type = 'image/jpeg'
    elif ext.lower() in ['.png']:
        content_type = 'image/png'
    elif ext.lower() in ['.mp4', '.mov', '.avi', '.mkv']:
        content_type = 'video/mp4'
    elif ext.lower() in ['.srt']:
        content_type = 'text/plain'
    
    # Upload the file without public-read ACL
    s3.upload_file(
        file_path, 
        s3_bucket, 
        s3_key, 
        ExtraArgs={
            'ContentType': content_type
        }
    )
    
    # Generate a pre-signed URL
    presigned_url = s3.generate_presigned_url(
        'get_object',
        Params={'Bucket': s3_bucket, 'Key': s3_key},
        ExpiresIn=expiration
    )
    
    return presigned_url

def generate_caption(image_url, client):
    caption_system_prompt = '''
You are an assistant that generates concise captions for images. These captions will be embedded and stored so 
people can semantically search for scenes. Ensure your captions include:
- Physical descriptions of people
- Identify and name key actors
- Descriptions of key scene objects such as their color
- The mood of the scene
- The actions or activities taking place in the scene
- Describe the quality of the image and suitability for promotional material
- Describe the angle and depth of the images (e.g., zoomed in close-up, zoomed out, etc.)
'''

    # Generate caption using GPT-4o for the provided image URL
    try:
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": caption_system_prompt}
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Generate a caption for this image:"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url,
                            }
                        },
                    ],
                }
            ],
        )

        # Access the message content directly from 'choices'
        caption = completion.choices[0].message.content.strip()

        return caption
    except Exception as e:
        st.error(f"Error generating caption: {e}")
        return None

def get_embedding(text, client, model="text-embedding-ada-002"):
    # Replace newline characters with spaces to ensure single-line input
    text = text.replace("\n", " ")
    
    # Create the embedding by passing a list containing the text
    try:
        response = client.embeddings.create(
            input=[text],
            model=model
        )
        # Access the embedding using attribute-based access
        return response.data[0].embedding
    except Exception as e:
        st.error(f"Error generating embedding: {e}")
        return None

def search_scenes(prompt, df_scenes, client, top_n=5):
    prompt_embedding = get_embedding(prompt, client)
    if prompt_embedding is None:
        return pd.DataFrame()  # Return empty DataFrame if embedding failed
    
    df_scenes['similarity'] = df_scenes['embedding'].apply(
        lambda x: cosine_similarity([x], [prompt_embedding])[0][0]
    )
    results = df_scenes.sort_values('similarity', ascending=False).head(top_n)
    return results

if __name__ == "__main__":
    main()
