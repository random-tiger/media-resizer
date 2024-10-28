# app/utils.py

import os
import requests
import boto3
from PIL import Image
import streamlit as st
import openai
import logging
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(name)s:%(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def is_url_accessible(url, method='HEAD', timeout=10):
    """
    Checks if a URL is accessible.

    Parameters:
        url (str): The URL to check.
        method (str): HTTP method to use ('HEAD' or 'GET').
        timeout (int): Timeout in seconds for the request.

    Returns:
        bool: True if accessible, False otherwise.
    """
    try:
        if method.upper() == 'HEAD':
            response = requests.head(url, timeout=timeout)
        else:
            response = requests.get(url, timeout=timeout)
        accessible = response.status_code == 200
        logger.info(f"URL accessibility check for {url}: {accessible}")
        return accessible
    except requests.RequestException as e:
        logger.error(f"URL accessibility check failed for {url}: {e}")
        return False

def upload_file_to_s3(file_path, s3_client, bucket_name, region, folder='files', acl='public-read', content_type='application/octet-stream'):
    """
    Uploads a file to AWS S3 and returns its public URL.

    Parameters:
        file_path (str): Path to the file to upload.
        s3_client (boto3.client): Boto3 S3 client.
        bucket_name (str): Name of the S3 bucket.
        region (str): AWS region where the bucket is located.
        folder (str): S3 folder prefix.
        acl (str): Access control list policy (default is 'public-read').
        content_type (str): MIME type of the file.

    Returns:
        str: Public URL of the uploaded file.
    """
    file_name = os.path.basename(file_path)
    s3_key = f"{folder}/{file_name}"
    
    try:
        s3_client.upload_file(
            file_path, 
            bucket_name, 
            s3_key, 
            ExtraArgs={
                'ACL': acl,
                'ContentType': content_type
            }
        )
        logger.info(f"Uploaded {file_path} to s3://{bucket_name}/{s3_key}")
    except Exception as e:
        logger.error(f"Failed to upload {file_path} to S3: {e}")
        raise e
    
    file_url = f"https://{bucket_name}.s3.{region}.amazonaws.com/{s3_key}"
    logger.info(f"Generated S3 URL: {file_url}")
    return file_url

def verify_image(file_path):
    """
    Verifies that the image file is not corrupted.

    Parameters:
        file_path (str): Path to the image file.

    Returns:
        bool: True if the image is valid, False otherwise.
    """
    try:
        with Image.open(file_path) as img:
            img.verify()  # Verify that it is, in fact, an image
        logger.info(f"Image verification passed for {file_path}")
        return True
    except (IOError, SyntaxError) as e:
        logger.error(f"Image verification failed for {file_path}: {e}")
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Corrupted image file {file_path} removed.")
        return False

def clean_up_files(file_paths):
    """
    Deletes a list of files if they exist.

    Parameters:
        file_paths (list of str): List of file paths to delete.
    """
    for file_path in file_paths:
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"Deleted temporary file: {file_path}")
            except Exception as e:
                logger.error(f"Failed to delete {file_path}: {e}")

def generate_caption(image_url, model="gpt-4"):
    """
    Generates a caption for an image using OpenAI's ChatCompletion API.

    Parameters:
        image_url (str): URL of the image.
        model (str): OpenAI model to use (default is "gpt-4").

    Returns:
        str or None: Generated caption or None if failed.
    """
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

    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": caption_system_prompt
                },
                {
                    "role": "user",
                    "content": f"Generate a caption for this image: {image_url}"
                }
            ],
            max_tokens=150,
            n=1,
            stop=None,
            temperature=0.7,
        )
        caption = response.choices[0].message.content.strip()
        logger.info(f"Generated caption for {image_url}: {caption}")
        return caption
    except openai.error.OpenAIError as e:
        st.error(f"OpenAI API Error: {e}")
        logger.error(f"OpenAI API Error while generating caption: {e}")
        return None
    except Exception as e:
        st.error(f"Unexpected Error: {e}")
        logger.error(f"Unexpected Error while generating caption: {e}")
        return None

def get_embedding(text, model="text-embedding-ada-002"):
    """
    Generates an embedding vector for the given text using OpenAI's Embeddings API.

    Parameters:
        text (str): The text to embed.
        model (str): The OpenAI model to use for embeddings.

    Returns:
        list or None: Embedding vector or None if failed.
    """
    try:
        response = openai.Embedding.create(
            input=text,
            model=model
        )
        embedding = response['data'][0]['embedding']
        logger.info(f"Generated embedding for text: {text[:30]}...")
        return embedding
    except openai.error.OpenAIError as e:
        st.error(f"OpenAI API Error: {e}")
        logger.error(f"OpenAI API Error while generating embedding: {e}")
        return None
    except Exception as e:
        st.error(f"Unexpected Error: {e}")
        logger.error(f"Unexpected Error while generating embedding: {e}")
        return None

def generate_unique_filename(original_filename):
    """
    Generates a unique filename by appending a UUID before the file extension.

    Parameters:
        original_filename (str): Original name of the file.

    Returns:
        str: Unique filename.
    """
    base, ext = os.path.splitext(original_filename)
    unique_name = f"{base}_{uuid.uuid4().hex}{ext}"
    logger.info(f"Generated unique filename: {unique_name}")
    return unique_name

def extract_frame_from_video(video_path, output_path, frame_time=1.0):
    """
    Extracts a frame from a video at a specified time.

    Parameters:
        video_path (str): Path to the video file.
        output_path (str): Path to save the extracted frame image.
        frame_time (float): Time (in seconds) at which to extract the frame.

    Returns:
        bool: True if extraction was successful, False otherwise.
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video file {video_path}")
        return False
    
    cap.set(cv2.CAP_PROP_POS_MSEC, frame_time * 1000)  # Set position in milliseconds
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        cv2.imwrite(output_path, frame)
        if verify_image(output_path):
            logger.info(f"Extracted frame at {frame_time} seconds to {output_path}")
            return True
        else:
            logger.error(f"Extracted frame is corrupted: {output_path}")
            return False
    else:
        logger.error(f"Failed to read frame at {frame_time} seconds from {video_path}")
        return False

def generate_presigned_url(s3_client, bucket_name, s3_key, expiration=3600):
    """
    Generates a presigned URL for an S3 object.

    Parameters:
        s3_client (boto3.client): Boto3 S3 client.
        bucket_name (str): Name of the S3 bucket.
        s3_key (str): S3 object key.
        expiration (int): Time in seconds for the presigned URL to remain valid.

    Returns:
        str or None: Presigned URL if successful, None otherwise.
    """
    try:
        url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': bucket_name, 'Key': s3_key},
            ExpiresIn=expiration
        )
        logger.info(f"Generated presigned URL for s3://{bucket_name}/{s3_key}")
        return url
    except Exception as e:
        logger.error(f"Failed to generate presigned URL for s3://{bucket_name}/{s3_key}: {e}")
        return None
