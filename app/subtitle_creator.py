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
    st.header("Subtitle Creator")
    
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
        temp_video_path = tfile.name
        
        # Display the video
        st.video(temp_video_path)
        
        # Extract audio from video
        st.write("Extracting audio from the video...")
        try:
            audio_file_path = extract_audio(temp_video_path)
            st.write("Audio extracted successfully.")
        except Exception as e:
            st.error(f"Error extracting audio: {e}")
            clean_up_files([temp_video_path])
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
            clean_up_files([audio_file_path, temp_video_path])
            return
        
        # Create a placeholder to show progress
        progress_bar = st.progress(0)
        total_languages = len(selected_languages)
        current_progress = 0
        
        # Dictionary to store subtitles
        subtitles = {}
        
        for language in selected_languages:
            code = language_codes.get(language)
            if not code:
                st.error(f"Unsupported language: {language}")
                continue
            st.write(f"Transcribing audio to {language}...")
            try:
                subtitle_content = generate_subtitles(audio_file_path, code, client)
                if subtitle_content:
                    st.write(f"Subtitles in {language} generated successfully.")
                    subtitles[language] = subtitle_content
            except Exception as e:
                st.error(f"Error generating subtitles in {language}: {e}")
            current_progress += 1
            progress_bar.progress(current_progress / total_languages)
        
        if not subtitles:
            st.warning("No subtitles were generated.")
            clean_up_files([audio_file_path, temp_video_path])
            return
        
        # Provide download buttons for subtitles
        st.write("### Download Subtitles")
        for language, subtitle_content in subtitles.items():
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
            language_to_embed = st.selectbox("Select language to embed into video", list(subtitles.keys()))
            st.write(f"Embedding {language_to_embed} subtitles into the video...")
            try:
                subtitled_video_path = embed_subtitles_into_video(temp_video_path, subtitles[language_to_embed], language_to_embed)
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
        clean_up_files([audio_file_path, temp_video_path, subtitled_video_path if embed_option == "Yes" else None])

def extract_audio(video_file_path):
    # Use moviepy to extract audio
    with mp.VideoFileClip(video_file_path) as clip:
        audio = clip.audio
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as audio_file:
            audio.write_audiofile(audio_file.name)
            audio_path = audio_file.name
    return audio_path

def generate_subtitles(audio_file_path, language_code):
    try:
        response = openai.Audio.transcriptions.create(
            file=open(audio_file_path, "rb"),
            model="whisper-1",
            response_format="srt",  # String-based response
            language=language_code
        )
        transcription = response  # response is a string in SRT format
        return transcription
    except openai.error.OpenAIError as e:
        st.error(f"OpenAI API Error: {e}")
        logger.error(f"OpenAI API Error during transcription: {e}")
        return None
    except Exception as e:
        st.error(f"Unexpected Error: {e}")
        logger.error(f"Unexpected Error during transcription: {e}")
        return None

def embed_subtitles_into_video(video_file_path, subtitle_content, language):
    # Save the subtitles to a .srt file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.srt', mode='w', encoding='utf-8') as subtitle_file:
        subtitle_file.write(subtitle_content)
        subtitle_path = subtitle_file.name
    
    # Output video file path
    output_video_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    output_video_path = output_video_file.name
    output_video_file.close()
    
    # Handle paths differently based on OS
    if platform.system() == 'Windows':
        # For Windows, need to escape backslashes and colons
        subtitle_path_escaped = subtitle_path.replace('\\', '\\\\').replace(':', '\\:')
        vf_arg = f"subtitles={subtitle_path_escaped}"
    else:
        vf_arg = f"subtitles='{subtitle_path}'"
    
    # Use ffmpeg to embed subtitles
    cmd = [
        'ffmpeg',
        '-i', video_file_path,
        '-vf', vf_arg,
        '-c:a', 'copy',
        output_video_path,
        '-y'  # Overwrite output file if it exists
    ]
    
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    if result.returncode != 0:
        error_message = result.stderr.decode()
        raise Exception(f"ffmpeg error: {error_message}")
    
    # Optionally, remove the subtitle file after embedding
    os.remove(subtitle_path)
    
    return output_video_path
