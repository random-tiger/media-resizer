# app/subtitle_creator.py

import streamlit as st
import moviepy.editor as mp
import tempfile
import os
import subprocess
import platform
from utils import clean_up_files
from openai import OpenAI

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
