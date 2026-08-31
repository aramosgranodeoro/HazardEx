import cv2
import math
import base64
from PIL import Image, ImageDraw
import io
import cv2
import tempfile
import os
from datetime import datetime
import json

MESES_ES = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]

import subprocess

def transcode_to_h264(video_bytes: bytes) -> bytes:
    """Transcodifica un vídeo a H.264/AAC en MP4, compatible con navegadores."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_in:
        tmp_in.write(video_bytes)
        input_path = tmp_in.name

    output_path = input_path.replace(".mp4", "_h264.mp4")

    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", input_path,
                "-c:v", "libx264",
                "-preset", "fast",
                "-c:a", "aac",
                "-movflags", "+faststart",
                output_path,
            ],
            capture_output=True,
            timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg falló: {result.stderr.decode(errors='ignore')}")

        with open(output_path, "rb") as f:
            return f.read()
    finally:
        os.unlink(input_path)
        if os.path.exists(output_path):
            os.unlink(output_path)


def frame_to_jpeg_bytes(frame: Image.Image) -> bytes:
    buf = io.BytesIO()
    frame.convert("RGB").save(buf, format="JPEG", quality=90)
    return buf.getvalue()

def get_month() -> str:
    now = datetime.now()
    return f"{now.day} {MESES_ES[now.month - 1]}"

def generic_media_title(media_type: str) -> str:
    """Genera un título genérico para un archivo multimedia."""
    etiqueta = "Vídeo" if media_type == "video" else "Imagen"
    return f"{etiqueta} - {get_month()}"

def truncate_title(text: str, max_words: int = 8) -> str:
    words = text.strip().split()
    if not words:
        return "Nueva conversación"
    if len(words) <= max_words:
        return text.strip()
    return " ".join(words[:max_words]) + "..."

def byte_to_base64(image_bytes):
    """Convierte bytes de una imagen en una cadena base64."""
    return base64.b64encode(image_bytes).decode('utf-8')

def pil_to_base64(image: Image.Image) -> str:
    """Convierte una imagen PIL en una cadena base64."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def frame_to_jpeg_bytes(frame: Image.Image) -> bytes:
    """Convierte un frame PIL en bytes JPEG."""
    buf = io.BytesIO()
    frame.convert("RGB").save(buf, format="JPEG", quality=90)
    return buf.getvalue()

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
        os.unlink(tmp_path) 

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

    # grid.show()
    return pil_to_base64(grid)

def build_analysis_text(result: dict) -> str:
    """Convierte el dict de módulos especializados en texto legible.
    
    Soporta varios formatos de salida:
    - Detección (fuego, armas): {"detected": bool, "detections": [...]}
    - VLM tipo respuesta (news, etc.): {"answer": str, "confidence": float} o string JSON de eso
    - Clasificación (violencia, tráfico): {"predicted_category": str, "confidence": float, "description": str}
    """
    partes = []

    if "general" in result:
        general = result.pop("general")
        if isinstance(general, dict):
            desc = general.get("description", "")
            top_cats = sorted(
                general.get("predicted_categories", []),
                key=lambda c: c.get("confidence", 0),
                reverse=True
            )
            top_text = ", ".join(f"{c['category']} ({c['confidence']:.2f})" for c in top_cats[:3])
            partes.append(f"Análisis general del contenido: {desc} Categorías más probables: {top_text}.")

    for categoria, data in result.items():
        # Si viene como string, intentar parsear JSON
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except (json.JSONDecodeError, TypeError):
                partes.append(f"{categoria}: {data}")
                continue

        if not isinstance(data, dict):
            partes.append(f"{categoria}: {data}")
            continue

        # Formato 1: detección con bounding boxes (fuego, armas)
        if "detected" in data:
            detected = data.get("detected", False)
            if not detected:
                partes.append(f"{categoria}: no detectado")
                continue
            detections = data.get("detections", [])
            if detections:
                confs = ", ".join(f"{d['confidence']:.2f}" for d in detections)
                partes.append(
                    f"{categoria}: detectado ({len(detections)} detección(es), "
                    f"confianza: {confs})"
                )
            else:
                partes.append(f"{categoria}: detectado")
            continue

        # Formato 2: respuesta VLM tipo pregunta/respuesta (news, OCR, etc.)
        if "answer" in data:
            confidence = data.get("confidence")
            conf_text = f" (confianza: {confidence:.2f})" if isinstance(confidence, (int, float)) else ""
            partes.append(f"{categoria}: {data['answer']}{conf_text}")
            continue

        # Formato 3: clasificación con categoría predicha (violencia, tráfico)
        if "predicted_category" in data:
            confidence = data.get("confidence", 0.0)
            description = data.get("description", "")
            partes.append(
                f"{categoria}: {data['predicted_category']} "
                f"(confianza: {confidence:.2f}) - {description}"
            )
            continue

        # Formato 4: múltiples categorías predichas (predicted_categories)
        if "predicted_categories" in data:
            top = max(data["predicted_categories"], key=lambda c: c.get("confidence", 0))
            description = data.get("description", "")
            partes.append(
                f"{categoria}: {top['category']} "
                f"(confianza: {top['confidence']:.2f}) - {description}"
            )
            continue

        # Fallback: formato no reconocido, lo mostramos tal cual
        partes.append(f"{categoria}: {data}")

    if not partes:
        return "Sin resultados de los módulos especializados."

    return "; ".join(partes)

def resize_image(image_bytes: bytes, max_size: int = 1024) -> bytes:
    """
    Redimensiona una imagen manteniendo su relación de aspecto.

    Args:
        image_bytes: Imagen en formato bytes.
        max_size: Tamaño máximo permitido para ancho/alto.

    Returns:
        Imagen redimensionada en formato JPEG como bytes.
    """

    image = Image.open(io.BytesIO(image_bytes))

    # Convertir a RGB para evitar problemas con PNG/RGBA/etc.
    if image.mode != "RGB":
        image = image.convert("RGB")

    width, height = image.size

    # Si ya cumple el tamaño máximo, devolverla igualmente en JPEG
    if width <= max_size and height <= max_size:
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=90)
        return output.getvalue()

    # Mantener proporción
    scale = max_size / max(width, height)

    new_width = int(width * scale)
    new_height = int(height * scale)

    image = image.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS
    )

    output = io.BytesIO()
    image.save(output, format="JPEG", quality=90)

    return output.getvalue()

