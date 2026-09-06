import base64
from dataclasses import dataclass
from typing import Optional

from app.storage.minio_client import upload_media
from app.triage.utils import (
    frame_to_jpeg_bytes,
    extract_frames,
    transcode_to_h264,
    frames_a_grid,
    build_annotated_image,
)


@dataclass
class UploadedMedia:
    media_id: str
    media_type: str  # "photo" | "video"
    annotated_media_id: Optional[str]


def upload_annotated_image(thread_id: str, media_id: str, image, result) -> Optional[str]:
    """Genera y sube la imagen anotada con bounding boxes, si existe."""
    annotated_bytes = build_annotated_image(image, result)
    if not annotated_bytes:
        return None

    annotated_media_id = f"{media_id}_annotated"
    upload_media(thread_id, annotated_media_id, annotated_bytes, "image/jpeg")
    return annotated_media_id


def upload_video_assets(thread_id: str, media_id: str, media_bytes: bytes) -> None:
    """Transcodifica el vídeo y genera/sube thumbnail + grid de frames."""
    h264_bytes = transcode_to_h264(media_bytes)
    upload_media(thread_id, media_id, h264_bytes, "video/mp4")

    frames = extract_frames(media_bytes)
    thumb_bytes = frame_to_jpeg_bytes(frames[0])
    upload_media(thread_id, f"{media_id}_thumb", thumb_bytes, "image/jpeg")

    grid_b64 = frames_a_grid(frames)
    grid_bytes = base64.b64decode(grid_b64)
    upload_media(thread_id, f"{media_id}_grid", grid_bytes, "image/jpeg")


def process_and_upload_media(
    thread_id: str,
    media_id: str,
    media_bytes: bytes,
    content_type: str,
    image,
    result,
) -> UploadedMedia:
    """Sube el archivo original (foto o vídeo + derivados) y la imagen anotada si aplica."""
    annotated_media_id = upload_annotated_image(thread_id, media_id, image, result)

    media_type = "video" if content_type.startswith("video") else "photo"
    if media_type == "video":
        upload_video_assets(thread_id, media_id, media_bytes)
    else:
        upload_media(thread_id, media_id, media_bytes, content_type)

    return UploadedMedia(media_id=media_id, media_type=media_type, annotated_media_id=annotated_media_id)