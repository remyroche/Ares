import boto3
from botocore.exceptions import ClientError
from app.config import settings
import os


class StorageService:
    def __init__(self):
        if settings.DEMO_MODE:
            self.s3 = None
            self.bucket = "demo-local"
            os.makedirs("exports/storage", exist_ok=True)
            return

        self.s3 = boto3.client(
            "s3",
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            region_name=settings.AWS_REGION,
            endpoint_url=settings.S3_ENDPOINT_URL if settings.S3_ENDPOINT_URL else None,
        )
        self.bucket = settings.S3_BUCKET

    def upload_bytes(
        self, data: bytes, key: str, content_type: str = "application/octet-stream"
    ):
        if settings.DEMO_MODE:
            path = os.path.join("exports/storage", key)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as f:
                f.write(data)
            return
        self.s3.put_object(
            Bucket=self.bucket, Key=key, Body=data, ContentType=content_type
        )

    def upload_file(
        self, file_path: str, key: str, content_type: str = "application/octet-stream"
    ):
        if settings.DEMO_MODE:
            path = os.path.join("exports/storage", key)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            import shutil

            shutil.copy2(file_path, path)
            return
        self.s3.upload_file(
            file_path, self.bucket, key, ExtraArgs={"ContentType": content_type}
        )

    def download_file(self, key: str, dest_path: str):
        if settings.DEMO_MODE:
            path = os.path.join("exports/storage", key)
            import shutil

            shutil.copy2(path, dest_path)
            return
        self.s3.download_file(self.bucket, key, dest_path)

    def exists(self, key: str) -> bool:
        if settings.DEMO_MODE:
            return os.path.exists(os.path.join("exports/storage", key))
        try:
            self.s3.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError:
            return False

    def signed_url(self, key: str, expires_sec: int = 3600) -> str:
        if settings.DEMO_MODE:
            return f"file://{os.path.abspath(os.path.join('exports/storage', key))}"
        return self.s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": key},
            ExpiresIn=expires_sec,
        )

    def compute_key(self, entity_type: str, entity_id: str, filename: str) -> str:
        return f"{settings.STORAGE_PREFIX}/{entity_type}/{entity_id}/{filename}"
