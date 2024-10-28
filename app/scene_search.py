# app/scene_search.py

import streamlit as st
import tempfile
import os
import subprocess
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import boto3
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector
from utils import upload_file_to_s3, is_url_accessible, clean_up_files
from utils import generate_caption, get_embedding

def scene_search_mode():
    st.header("Scene Search")
    
    # Check for AWS credentials and OpenAI API key in st.secrets
    required_secrets = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_S3_BUCKET_NAME", "OPENAI_API_KEY"]
    if not all(secret in st.secrets for secret in required_secrets):
        st.error("AWS credentials or OpenAI API key not found in st.secrets. Please add them to your Streamlit secrets.")
        return
    
    aws_access_key_id = st.secrets["AWS_ACCESS_KEY_ID"]
    aws_secret_access_key = st.secrets["AWS_SECRET_ACCESS_KEY"]
    s3_bucket_name = st.secrets["AWS_S3_BUCKET_NAME"]
    s3_region = st.secrets.get("AWS_REGION", "us-east-1")
    openai_api_key = st.secrets["OPENAI_API_KEY"]
    
    # Initialize S3 client
    try:
        s3 = boto3.client(
            's3',
            region_name=s3_region,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key
        )
    except Exception as e:
        st.error(f"Error initializing S3 client: {e}")
        return
    
    # Initialize OpenAI client
    try:
        client = OpenAI(api_key=openai_api_key)
    except Exception as e:
        st.error(f"Error initializing OpenAI client: {e}")
        return
    
    # Prompt user to upload a video file
    uploaded_video = st.file_uploader("Upload a video file for scene search", type=["mp4", "avi", "mov", "mkv"])
    if uploaded_video is not None:
        # Save the uploaded video to a temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_video.read())
        tfile.flush()
        temp_video_path = tfile.name
        
        # Display the video
        st.video(temp_video_path)
        
        # Extract scenes from the video
        st.write("Extracting scenes from the video...")
        try:
            scene_list = extract_scenes(temp_video_path)
            st.write(f"Extracted {len(scene_list)} scenes.")
        except Exception as e:
            st.error(f"Error extracting scenes: {e}")
            clean_up_files([temp_video_path])
            return
        
        if not scene_list:
            st.warning("No scenes were detected in the video.")
            clean_up_files([temp_video_path])
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
            clean_up_files([temp_video_path] + scene_list)
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
        clean_up_files([temp_video_path] + scene_list + df_scenes['frame_filename'].tolist())
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
            if os.path.exists(output_filepath):
                scene_filenames.append(output_filepath)
        return scene_filenames

def process_scene(scene_filename, s3, s3_bucket_name, s3_region, client):
    # Upload scene clip to S3 and get the URL
    try:
        scene_url = upload_file_to_s3(scene_filename, s3, s3_bucket_name, s3_region, folder='scenes', content_type='video/mp4')
    except Exception as e:
        st.error(f"Error uploading scene {scene_filename} to S3: {e}")
        return None
    
    # Extract a frame from the scene
    frame_filename = extract_frame_from_scene(scene_filename)
    if frame_filename is None:
        return None

    # Upload the frame image to S3 and get the URL
    try:
        frame_url = upload_file_to_s3(frame_filename, s3, s3_bucket_name, s3_region, folder='frames', content_type='image/jpeg')
    except Exception as e:
        st.error(f"Error uploading frame {frame_filename} to S3: {e}")
        return None
    
    # Verify if the image URL is accessible
    if not is_url_accessible(frame_url):
        st.error(f"Generated image URL is not accessible: {frame_url}")
        return None
    
    # Generate caption using GPT-4 with the frame URL
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
        'frame_filename': frame_filename,
        'frame_url': frame_url,
        'caption': caption,
        'embedding': embedding
    }

def extract_frame_from_scene(scene_filename):
    import cv2
    from PIL import Image
    from .utils import verify_image
    
    cap = cv2.VideoCapture(scene_filename)
    if not cap.isOpened():
        st.error(f"Failed to open video file {scene_filename}")
        return None
    ret, frame = cap.read()
    cap.release()
    if ret:
        frame_filename = scene_filename + "_frame.jpg"
        
        # Resize the frame to reduce size and prevent timeout
        desired_width = 800  # Adjust as needed
        height, width, channels = frame.shape
        if height == 0:
            st.error(f"Invalid frame dimensions for {scene_filename}")
            return None
        aspect_ratio = width / height
        desired_height = int(desired_width / aspect_ratio)
        resized_frame = cv2.resize(frame, (desired_width, desired_height))
        
        cv2.imwrite(frame_filename, resized_frame)
        
        # Verify the saved image is not corrupted
        if verify_image(frame_filename):
            return frame_filename
        else:
            return None
    else:
        st.error(f"Failed to read frame from {scene_filename}")
        return None

def search_scenes(prompt, df_scenes, client, top_n=5):
    prompt_embedding = get_embedding(prompt, client)
    if prompt_embedding is None:
        st.warning("Failed to generate embedding for the prompt.")
        return pd.DataFrame()  # Return empty DataFrame if embedding failed
    
    df_scenes['similarity'] = df_scenes['embedding'].apply(
        lambda x: cosine_similarity([x], [prompt_embedding])[0][0]
    )
    results = df_scenes.sort_values('similarity', ascending=False).head(top_n)
    return results
