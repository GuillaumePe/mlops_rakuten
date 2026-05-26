"""
Upload/download des logs entrypoint vers/depuis R2.

Côté pod (upload) :
    python scripts/r2_logs.py upload /path/to/local.log job_xxx.log

Côté local (download) :
    python scripts/r2_logs.py download job_xxx.log /tmp/job_xxx.log
    python scripts/r2_logs.py list                # liste les logs récents
"""
import os
import sys
import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv
load_dotenv()

R2_BUCKET = os.environ.get("R2_BUCKET_NAME", "rakuten-mlops-dvc")
R2_LOG_PREFIX = "logs/"


def get_client():
    endpoint = os.environ["R2_ENDPOINT_URL"]
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )


def upload(local_path: str, remote_name: str):
    s3 = get_client()
    key = R2_LOG_PREFIX + remote_name
    s3.upload_file(local_path, R2_BUCKET, key)
    print(f"[r2_logs] Uploaded -> s3://{R2_BUCKET}/{key}")


def download(remote_name: str, local_path: str):
    s3 = get_client()
    key = R2_LOG_PREFIX + remote_name
    s3.download_file(R2_BUCKET, key, local_path)
    print(f"[r2_logs] Downloaded s3://{R2_BUCKET}/{key} -> {local_path}")


def list_logs():
    s3 = get_client()
    resp = s3.list_objects_v2(Bucket=R2_BUCKET, Prefix=R2_LOG_PREFIX)
    objects = sorted(resp.get("Contents", []), key=lambda o: o["LastModified"], reverse=True)
    print(f"{'Last modified':<25}  {'Size':>10}  Key")
    print("-" * 80)
    for obj in objects[:30]:
        size_mb = obj["Size"] / 1024
        print(f"{obj['LastModified'].isoformat():<25}  {size_mb:>8.1f}KB  {obj['Key']}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else None
    try:
        if cmd == "upload":
            upload(sys.argv[2], sys.argv[3])
        elif cmd == "download":
            download(sys.argv[2], sys.argv[3])
        elif cmd == "list":
            list_logs()
        else:
            print(__doc__)
            sys.exit(1)
    except ClientError as e:
        print(f"[r2_logs] ERROR: {e}")
        sys.exit(2)