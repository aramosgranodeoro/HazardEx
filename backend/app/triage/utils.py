import cv2
import math
import base64
from PIL import Image, ImageDraw
import io
import cv2
import tempfile
import os

def byte_to_base64(image_bytes):
    """Convierte bytes de una imagen en una cadena base64."""
    return base64.b64encode(image_bytes).decode('utf-8')

def pil_to_base64(image: Image.Image) -> str:
    """Convierte una imagen PIL en una cadena base64."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def resize_image(image_bytes: bytes, size=(224, 224)) -> bytes:
    """Redimensionar imagen para que tenga un tamaño máximo de 224x224 píxeles."""
    image = Image.open(io.BytesIO(image_bytes))
    image = image.resize(size)
    return pil_to_base64(image)

def extract_frames(video_bytes: bytes, n: int = 9) -> list[Image.Image]:
    """
    Recibe los bytes de un vídeo MP4 y extrae n frames con más peso en el
    último 50% (donde suelen ocurrir los accidentes).
    Devuelve lista de PIL Images redimensionadas a 224x168.
    """
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    try:
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise IOError("No se pudo abrir el vídeo (bytes corruptos o formato no soportado)")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            raise ValueError("Vídeo sin frames detectables")

        n_ini = max(1, int(n * 0.3))
        n_fin = n - n_ini
        idx_ini = [int(total_frames * i / (n_ini * 3)) for i in range(n_ini)]
        idx_fin = [int(total_frames * (0.5 + 0.5 * i / n_fin)) for i in range(n_fin)]
        indices = sorted(set(idx_ini + idx_fin))
        indices = [min(idx, total_frames - 1) for idx in indices]

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb).resize((224, 168))
                frames.append(pil_img)
        cap.release()

        if not frames:
            raise ValueError("No se extrajeron frames del vídeo")

        return frames

    finally:
        os.unlink(tmp_path)  # limpieza garantizada, incluso si algo falla arriba


# ── Grid de frames ────────────────────────────────────────────────────────────

def frames_a_grid(frames, cols=3):
    """
    Dibuja un grid de frames con timestamps y devuelve la imagen resultante.
    """
    n    = len(frames)
    rows = math.ceil(n / cols)
    w, h = frames[0].size
    grid = Image.new("RGB", (w * cols, h * rows), color=(0, 0, 0))
    for i, frame in enumerate(frames):
        frame = frame.copy()
        draw  = ImageDraw.Draw(frame)
        draw.rectangle([(0, 0), (55, 20)], fill=(0, 0, 0))
        draw.text((4, 4), f"t={i+1}/{n}", fill=(255, 255, 0))
        row, col = divmod(i, cols)
        grid.paste(frame, (col * w, row * h))

    grid.show()
    return pil_to_base64(grid)

# Unificar resultados de los módulos si necesario
def merge_results(results):
    # Implementar lógica de limpieza según sea necesario
    return results
