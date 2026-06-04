"""
Upload/download des logs entrypoint vers/depuis R2.

Usage simplifié (download par job_id) :
    python scripts/r2_logs.py g6el6kdji7aqd1      # dl le log du pod dans /tmp

Autres commandes :
    python scripts/r2_logs.py list                 # liste les logs récents
    python scripts/r2_logs.py upload /path/to/local.log job_xxx.log
    python scripts/r2_logs.py download job_xxx.log /tmp/job_xxx.log
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


def download_by_job_id(job_id: str):
    """Cherche le log contenant le job_id et le télécharge dans /tmp."""
    s3 = get_client()
    resp = s3.list_objects_v2(Bucket=R2_BUCKET, Prefix=R2_LOG_PREFIX)
    objects = sorted(resp.get("Contents", []), key=lambda o: o["LastModified"], reverse=True)

    matches = [obj for obj in objects if job_id in obj["Key"]]

    if not matches:
        print(f"[r2_logs] Aucun log trouvé pour job_id '{job_id}'")
        print(f"[r2_logs] Logs récents :")
        for obj in objects[:10]:
            print(f"  {obj['Key']}")
        sys.exit(1)

    for obj in matches:
        filename = obj["Key"].replace(R2_LOG_PREFIX, "")
        local_path = f"/tmp/{filename}"
        s3.download_file(R2_BUCKET, obj["Key"], local_path)
        print(f"[r2_logs] {local_path}")

        # Affiche le contenu directement
        #with open(local_path) as f:
        #    print(f.read())


def list_logs():
    s3 = get_client()
    resp = s3.list_objects_v2(Bucket=R2_BUCKET, Prefix=R2_LOG_PREFIX)
    objects = sorted(resp.get("Contents", []), key=lambda o: o["LastModified"], reverse=True)
    print(f"{'Last modified':<25}  {'Size':>10}  Key")
    print("-" * 80)
    for obj in objects[:30]:
        size_kb = obj["Size"] / 1024
        print(f"{obj['LastModified'].isoformat():<25}  {size_kb:>8.1f}KB  {obj['Key']}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else None
    try:
        if cmd == "upload":
            upload(sys.argv[2], sys.argv[3])
        elif cmd == "download":
            download(sys.argv[2], sys.argv[3])
        elif cmd == "list":
            list_logs()
        elif cmd is None:
            print(__doc__)
            sys.exit(1)
        else:
            # Argument unique = job_id → cherche et télécharge
            download_by_job_id(cmd)
    except ClientError as e:
        print(f"[r2_logs] ERROR: {e}")
        sys.exit(2)
