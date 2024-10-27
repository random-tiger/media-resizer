# app/utils.py

import os
import requests
import boto3
from PIL import Image
import streamlit as st

def is_url_accessible(url, method='GET', timeout=10):
    try:
        if method.upper() == 'HEAD':
            response = requests.head(url, timeout=timeout)
        else:
            response = requests.get(url, timeout=timeout)
        return response.status_code == 200
    except requests.RequestException:
        return False

def upload_file_to_s3(file_path, s3_client, bucket_name, region, folder='files', acl='public-read', content_type='application/octet-stream'):
    file_name = os.path.basename(file_path)
    s3_key = f"{folder}/{file_name}"
    
    # Upload the file with specified ACL and ContentType
    s3_client.upload_file(
        file_path, 
        bucket_name, 
        s3_key, 
        ExtraArgs={
            'ACL': acl,
            'ContentType': content_type
        }
    )
    
    # Generate a public URL
    file_url = f"https://{bucket_name}.s3.{region}.amazonaws.com/{s3_key}"
    return file_url

def verify_image(file_path):
    try:
        with Image.open(file_path) as img:
            img.verify()  # Verify that it is, in fact an image
        return True
    except Exception as e:
        st.error(f"Image verification failed: {e}")
        if os.path.exists(file_path):
            os.remove(file_path)
        return False

def clean_up_files(file_paths):
    for file_path in file_paths:
        if os.path.exists(file_path):
            os.remove(file_path)
