from fastapi import APIRouter, HTTPException, Request, Response
from minio import S3Error

from app import state
from app.storage.minio_client import get_media

router = APIRouter(tags=["media"])


@router.get("/media/{thread_id}/{media_id}")
async def get_media_endpoint(thread_id: str, media_id: str, request: Request):
    try:
        data = get_media(thread_id, media_id)
    except S3Error as e:
        if e.code == "NoSuchKey":
            raise HTTPException(status_code=404, detail="Media no encontrada")
        raise

    if data is None:
        raise HTTPException(status_code=404, detail="Media no encontrada")

    config = {"configurable": {"thread_id": thread_id}}
    conv_state = await state.agent.aget_state(config)
    available_media = conv_state.values.get("available_media", {}) if conv_state else {}
    media_type = available_media.get(media_id)
    content_type = "video/mp4" if media_type == "video" else "image/jpeg"

    file_size = len(data)
    range_header = request.headers.get("range")

    if range_header is None:
        return Response(
            content=data,
            media_type=content_type,
            headers={"Accept-Ranges": "bytes", "Content-Length": str(file_size)},
        )

    range_value = range_header.replace("bytes=", "").split("-")
    start = int(range_value[0]) if range_value[0] else 0
    end = int(range_value[1]) if len(range_value) > 1 and range_value[1] else file_size - 1
    end = min(end, file_size - 1)
    chunk = data[start:end + 1]

    headers = {
        "Content-Range": f"bytes {start}-{end}/{file_size}",
        "Accept-Ranges": "bytes",
        "Content-Length": str(len(chunk)),
    }
    return Response(content=chunk, status_code=206, media_type=content_type, headers=headers)