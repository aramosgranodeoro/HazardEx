import io
import json
from datetime import datetime
from minio.error import S3Error
from app.storage.minio_client import client, ensure_bucket, BUCKET_NAME
from minio.deleteobjects import DeleteObject

"""
Funciones para gestionar los metadatos de las conversaciones en MinIO.
"""

META_PREFIX = "conversations/"


def _meta_object_name(thread_id: str) -> str:
    return f"{META_PREFIX}{thread_id}.json"


def save_conversation_metadata(thread_id: str, title: str) -> dict:
    """Crea el JSON de metadatos de una conversación en MinIO (solo al crearla)."""
    ensure_bucket()
    metadata = {
        "thread_id": thread_id,
        "title": title,
        "created_at": datetime.now().isoformat(),
    }
    data = json.dumps(metadata, ensure_ascii=False).encode("utf-8")
    client.put_object(
        BUCKET_NAME,
        _meta_object_name(thread_id),
        data=io.BytesIO(data),
        length=len(data),
        content_type="application/json",
    )
    return metadata

def get_conversation_metadata(thread_id: str) -> dict | None:
    """Lee el JSON de metadatos de una conversación desde MinIO. None si no existe."""
    try:
        response = client.get_object(BUCKET_NAME, _meta_object_name(thread_id))
        try:
            return json.loads(response.read())
        finally:
            response.close()
            response.release_conn()
    except S3Error as e:
        if e.code == "NoSuchKey":
            return None
        raise

def list_conversations() -> list[dict]:
    ensure_bucket()
    conversations = []
    for obj in client.list_objects(BUCKET_NAME, prefix=META_PREFIX, recursive=True):
        response = client.get_object(BUCKET_NAME, obj.object_name)
        try:
            conversations.append(json.loads(response.read()))
        finally:
            response.close()
            response.release_conn()
    conversations.sort(key=lambda c: c["created_at"], reverse=True)
    return conversations

def delete_conversation_metadata(thread_id: str):
    """Borra todos los archivos asociados a una conversación."""

    try:
        prefix = f"{thread_id}/"

        objects = client.list_objects(
            BUCKET_NAME,
            prefix=prefix,
            recursive=True
        )

        delete_objects = (
            DeleteObject(obj.object_name)
            for obj in objects
        )

        errors = client.remove_objects(
            BUCKET_NAME,
            delete_objects
        )

        for error in errors:
            print(
                f"Error eliminando {error.object_name}: "
                f"{error.code} - {error.message}"
            )

    except S3Error as e:
        print(f"Error eliminando conversación {thread_id}: {e}")
        raise