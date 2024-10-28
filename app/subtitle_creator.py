# app/subtitle_creator.py

import streamlit as st
import moviepy.editor as mp
import tempfile
import os
import subprocess
import platform
import logging
from utils import clean_up_files
from openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO)

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
        try:
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
                    if not subtitle_content:
                        st.error(f"Failed to generate subtitles for {language}.")
                        continue  # Skip adding a download button for this language
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
                if subtitle_content:
                    srt_filename = f"subtitles_{language}.srt"
                    st.download_button(
                        label=f"Download {language} Subtitles (.srt)", 
                        data=subtitle_content, 
                        file_name=srt_filename,
                        mime="text/plain"
                    )
                else:
                    st.warning(f"No subtitles available for {language}.")
            
            # **Embed Subtitles into Video Section**
            st.write("### Embed Subtitles into Video")
            
            if not subtitles:
                st.warning("No subtitles available to embed.")
            else:
                # Allow user to select the language for embedding
                language_to_embed = st.selectbox(
                    "Select language to embed into video", 
                    list(subtitles.keys()),
                    key="embed_language_select"
                )
                
                # Add the Embed button
                if st.button("Embed Subtitles"):
                    st.write(f"Embedding {language_to_embed} subtitles into the video...")
                    try:
                        # Perform the embedding
                        subtitled_video_path = embed_subtitles_into_video(
                            tfile.name, 
                            subtitles[language_to_embed], 
                            language_to_embed
                        )
                        
                        # Display the subtitled video
                        st.video(subtitled_video_path)
                        
                        # Provide a download button for the subtitled video
                        with open(subtitled_video_path, 'rb') as f:
                            video_bytes = f.read()
                            st.download_button(
                                label=f"Download Video with {language_to_embed} Subtitles", 
                                data=video_bytes, 
                                file_name=f"video_with_{language_to_embed}_subtitles.mp4",
                                mime="video/mp4"
                            )
                    except Exception as e:
                        st.error(f"Error embedding subtitles: {e}")
                    finally:
                        # Optionally, clean up the subtitled video file
                        if os.path.exists(subtitled_video_path):
                            os.unlink(subtitled_video_path)
        
        finally:
            # Clean up temporary files
            if os.path.exists(tfile.name):
                os.unlink(tfile.name)
            if 'audio_file_path' in locals() and os.path.exists(audio_file_path):
                os.unlink(audio_file_path)
            # Note: subtitled_video_path is already cleaned up in the 'finally' block above
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
            if isinstance(response, str):
                return response  # Return the SRT string directly
            else:
                logging.error("Unexpected response format: %s", type(response))
                st.error("Received unexpected response format from the transcription API.")
                return None
        except Exception as e:
            logging.error(f"Error during transcription for language {language_code}: {e}")
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
    
    # Clean up the subtitle file
    if os.path.exists(subtitle_file.name):
        os.unlink(subtitle_file.name)
    
    # Return the path to the output video file
    return output_video_file.name
