import ollama
import re
from app.triage.models.model_manager import VLMModelManager, vlm_manager
from app.triage.models.fire_detector import FireSmokeDetector
from app.triage.models.weapon_detector import WeaponDetector
import base64
from io import BytesIO
from PIL import Image

# LLamadas a los módulos especializados para contexto iniciald del agente
MODEL = "qwen3.5:latest"

VIOLENCE_PROMPT = """
            # ROLE
            You are a security analyst reviewing surveillance footage.
         
            # CONTEXT
            You are looking at a grid of frames from a video in chronological
            order (t=1 is earliest). Analyze the PROGRESSION of events across
            frames, not just a single moment.

            # SIGNS OF VIOLENCE
            - People hitting, pushing, or grabbing each other
            - Aggressive or erratic movements between frames
            - People falling or being knocked down
            - Crowd suddenly gathering around a conflict

            # INSTRUCTIONS
            Choose exactly ONE label for "predicted_category":
            - "fight" if you see signs of violence
            - "non-fight" if you do NOT see signs of violence

            # OUTPUT FORMAT
            Respond with ONLY one valid JSON object and nothing else.

            Example of a valid response:
            {"predicted_category": "fight", "confidence": 0.87, "description": "Two people are seen exchanging blows starting at t=3."}
"""
ACCIDENT_PROMPT = """
# ROLE
            You are a traffic safety analyst reviewing dashcam footage.

            # CONTEXT
            You are looking at a grid of frames in chronological order (t=1 earliest).
            Analyze the PROGRESSION of events to detect a traffic accident.

            # SIGNS OF A CRASH
            - Sudden impact or collision between vehicles
            - Vehicle leaving the road or rolling over
            - Airbag deployment or broken glass
            - Abrupt stop or erratic movement between frames
            - Debris or damage visible in later frames
            - Vehicles too close to each other suddenly appearing in the same frame

            # INSTRUCTIONS
            Choose exactly ONE label for "predicted_category":
            - "crash" if you see signs of a traffic accident
            - "no_crash" if you do NOT see signs of a traffic accident

            # OUTPUT FORMAT
            Respond ONLY with a valid JSON object and nothing else.
            Example of a valid response:
            {
                "predicted_category": "crash",
                "confidence": 0.89,
                "description": "A vehicle has collided with another object in the road."
            }
"""

async def run_violence_module(media):
    result = await vlm_manager.generate(
        category="violence", 
        image=media, 
        prompt=VIOLENCE_PROMPT
    )
    return result

async def run_traffic_module(media):
    result = await vlm_manager.generate(
        category="traffic_accident", 
        image=media, 
        prompt=ACCIDENT_PROMPT
    )
    return result

def run_weapons_module(raw_image):
    # 1. Decodificar el string Base64 a bytes puros
    image_bytes = base64.b64decode(raw_image)

    # 2. Convertir los bytes a una imagen de PIL
    pil_image = Image.open(BytesIO(image_bytes))

    weapons_detector = WeaponDetector()
    weapons_detected = weapons_detector.predict(pil_image)
    return weapons_detected

def run_fire_smoke_module(raw_image):
    # 1. Decodificar el string Base64 a bytes puros
    image_bytes = base64.b64decode(raw_image)

    # 2. Convertir los bytes a una imagen de PIL
    pil_image = Image.open(BytesIO(image_bytes))

    fire_detector = FireSmokeDetector()
    fire_detected = fire_detector.predict(pil_image)
    return fire_detected

async def run_news_module(image):
    response = ollama.chat(
        model=MODEL,
        options={"temperature": 0.1, "num_predict": 4096, "num_ctx": 4096},
        messages=[{
            "role": "user",
            "content": """
            You are a OCR specialist. Do not think out loud. Do not use <tool_call> tags. Respond ONLY with the requested JSON.

            # OBJECTIVE
            Extract text from the image and explain the context of the chart or news article. If there is a tittle, include it in the answer.
            
            # OUTPUT FORMAT
            Respond ONLY with valid JSON:
            {
                "answer": "your answer here",
                "confidence": "a number between 0 and 1",
                "title": "the title of the news article, if any"
            }
            """,
            "images": [image]
        }]
    )
    raw = response["message"]["content"]
    raw = re.sub(r"<tool_call>.*?</tool_call>", "", raw, flags=re.DOTALL)
    raw = re.sub(r"<thinking>.*?</thinking>", "", raw, flags=re.DOTALL)
    print(f"Raw response from Ollama:\n{raw}\n")
    return raw.strip()
