# app/scene_search.py

from utils import upload_file_to_s3, is_url_accessible, clean_up_files
from utils import generate_caption, get_embedding
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

def scene_search_mode():
    st.header("Scene Search")
    
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

    # Upload the frame image to S3 and get the URL
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
    
    # Upload the file with public-read ACL
    s3.upload_file(
        file_path, 
        s3_bucket, 
        s3_key, 
        ExtraArgs={
            'ACL': 'public-read',
            'ContentType': content_type
        }
    )
    
    # Generate a public URL
    file_url = f"https://{s3_bucket}.s3.{s3_region}.amazonaws.com/{s3_key}"
    
    return file_url

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
