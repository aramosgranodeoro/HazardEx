# app/storage/minio_client.py
from minio import Minio
from minio.error import S3Error
import io

"""
Funciones para gestionar archivos en MinIO.
"""

BUCKET_NAME = "hazardex-media"

client = Minio(
    "localhost:9000",
    access_key="minioadmin",
    secret_key="minioadmin",
    secure=False,  # True si usas HTTPS
)

def ensure_bucket():
    if not client.bucket_exists(BUCKET_NAME):
        client.make_bucket(BUCKET_NAME)

def upload_media(thread_id: str, media_id: str, media_bytes: bytes, content_type: str):
    object_name = f"{thread_id}/{media_id}"
    client.put_object(
        BUCKET_NAME,
        object_name,
        io.BytesIO(media_bytes),
        length=len(media_bytes),
        content_type=content_type
    )
    return object_name


def get_media(thread_id: str, media_id: str) -> bytes:
    """Recupera los bytes del archivo asociado a un thread_id y media_id."""
    object_name = f"{thread_id}/{media_id}"
    response = client.get_object(BUCKET_NAME, object_name)
    try:
        return response.read()
    finally:
        response.close()
        response.release_conn()


def download_media(thread_id: str) -> bytes | None:
    """Recupera los bytes originales del archivo asociado a un thread_id."""
    try:
        response = client.get_object(BUCKET_NAME, thread_id)
        data = response.read()
        response.close()
        response.release_conn()
        return data
    except S3Error as e:
        if e.code == "NoSuchKey":
            return None
        raise

def delete_media(thread_id: str):
    """Borra el archivo (imagen/vídeo) asociado a un thread_id."""
    try:
        client.remove_object(BUCKET_NAME, thread_id)
    except S3Error as e:
        if e.code != "NoSuchKey":
            raise