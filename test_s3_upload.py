# %%
import os
import boto3

from dotenv import load_dotenv

load_dotenv()

# Initialize a session using Amazon S3 credentials
session = boto3.Session(
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_DEFAULT_REGION"),
)
s3 = session.resource("s3")

# Specify the S3 bucket
bucket_name = os.getenv("S3_BUCKET_NAME")

# Load images
image_path = "./data/images"
image_files = [f for f in os.listdir(image_path) if f.endswith(".jpeg")]

# Upload files to the specified S3 bucket
for image_file in image_files[:5]:
    file_path = os.path.join(image_path, image_file)
    with open(file_path, "rb") as file:
        binary_data = file.read()
        s3.Bucket(bucket_name).put_object(Key=image_file, Body=binary_data)
        print(f"Uploaded {image_file} to S3 bucket {bucket_name}")
