import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
import pandas as pd
import sys
import csv

if sys.version_info[0] < 3: 
    from StringIO import StringIO # Python 2.x
else:
    from io import StringIO # Python 3.x


def upload_to_aws(local_file, bucket, s3_file):
    
    session = boto3.session.Session()#boto3.Session(region_name = 'us-west-2')
    s3 = session.client('s3')
    try:
        s3.upload_file(local_file, bucket, s3_file)
        print("Upload Successful")
        return True
    except FileNotFoundError:
        print("The file was not found")
        return False
    except NoCredentialsError:
        print("Credentials not available")
        return False

def download_file_from_s3(bucket_name, s3_file_name, local_file_name):
    # Create an S3 client
    session = boto3.session.Session()#boto3.Session(region_name = 'us-west-2')
    s3 = session.client('s3')

    try:
        # Download the file from S3
        s3.download_file(bucket_name, s3_file_name, local_file_name)
        print(f"File {s3_file_name} downloaded from bucket {bucket_name} to {local_file_name}")

    except NoCredentialsError:
        print("Credentials not available")
    except PartialCredentialsError:
        print("Incomplete credentials provided")
    except Exception as e:
        print(f"An error occurred: {e}")

def read_contents_s3(bucket_name, s3_file_name):
    contents = []
    out=[]
    try:
        session = boto3.session.Session()#boto3.Session(region_name = 'us-west-2')
        s3 = session.client('s3')
        file = s3.list_objects(Bucket = bucket_name, Prefix='')
        contents = []
        for o in file.get('Contents'):
            data = s3.get_object(Bucket=bucket_name, Key=s3_file_name)
            contents = data['Body'].read()
        out = csv.reader(StringIO(contents.decode('utf-8')),dialect='excel')

    except NoCredentialsError:
        print("Credentials not available")
    except PartialCredentialsError:
        print("Incomplete credentials provided")
    except Exception as e:
        print(f"An error occurred: {e}")

    return out
