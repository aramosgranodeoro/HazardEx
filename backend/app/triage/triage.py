import ollama
import re
from .utils import byte_to_base64, extract_frames, frames_a_grid
from .models.modelsCalls import run_violence_module, run_traffic_module, run_weapons_module, run_fire_smoke_module, run_news_module
import json
from .models.model_manager import vlm_manager

MODEL = "qwen3.5:latest"

VIDEO_EXTENSIONS = ["mp4", "avi", "mov", "mkv", "flv", "wmv"]

PROMT_VIDEO = """
Analyze this grid of frames and estimate the probability of each safety category
# CONTEXT
    You are looking at a grid of frames in chronological order (t=1 earliest).
    Analyze the PROGRESSION of events to detect what is happening.
"""
PROMT_IMAGE = """
Analyze this image and estimate the probability of each safety category
"""

def classify_image(media, file_name):
    image = byte_to_base64(media)
    file_extension = file_name.split(".")[-1].lower()

    if file_extension in VIDEO_EXTENSIONS:
        aux = extract_frames(media)
        image = frames_a_grid(aux)
        prompt = PROMT_VIDEO
    else:
        prompt = PROMT_IMAGE

    response = ollama.chat(
        model=MODEL,
        options={"temperature": 0.1, "num_predict": 4096, "num_ctx": 8192},
        messages=[{
            "role": "user",
            "content": """
                You are a visual safety analyst. 
                
                {prompt}

                Possible categories:

                - violence: physical fight, assault, aggression between people
                - weapons: visible knife, gun, firearm, or dangerous weapon
                - fire: flames, smoke, burning objects or buildings
                - traffic_accident: traffic collision, car crash, road accident, cars too close, damaged vehicles, cars off the road
                - news: ANY graph, chart, table, infographic, or news article with data
                - normal: none of the above, everyday safe content

                Rules:
                1. Evaluate ALL categories and assign a confidence score between 0.00 and 1.00 to each one.
                2. The sum of all confidence scores does NOT need to be exactly 1.00, since multiple categories can be present at the same time.
                3. A single image or video can contain multiple safety categories (for example: accident + fire + weapons).
                4. If no safety category is detected, "normal" should have the highest confidence score.
                5. If you see ANY chart, graph, table, infographic, or data visualization, assign a high confidence score to "chart".
                6. Sort categories from highest to lowest confidence.
                7. Do not include explanations outside the JSON. 

                Respond ONLY with valid JSON:

                {
                    "predicted_categories": [
                        {
                            "category": "fire",
                            "confidence": 0.92
                        },
                        {
                            "category": "violence",
                            "confidence": 0.10
                        },
                        {
                            "category": "weapons",
                            "confidence": 0.05
                        },
                        {
                            "category": "traffic_accident",
                            "confidence": 0.20
                        },
                        {
                            "category": "news",
                            "confidence": 0.00
                        },
                        {
                            "category": "normal",
                            "confidence": 0.03
                        }
                    ],
                    "description": "One sentence explaining the reasoning behind the prediction."
                }

            """,
            "images": [image]
        }]
    )
    raw = response["message"]["content"]
    
    raw = re.sub(r"<tool_call>.*?</tool_call>", "", raw, flags=re.DOTALL)
    raw = re.sub(r"<thinking>.*?</thinking>", "", raw, flags=re.DOTALL)
    print(f"Raw response from Ollama:\n{raw}\n")
    return raw.strip(), image

# Llamar a módulos especializados para análisis de imagen o vídeo según el tipo de medio
async def run_specialized_modules(catergories, media):
    results = {} 
    categories = json.loads(catergories).get("predicted_categories", [])
    activated_categories = [c["category"] for c in categories if c["confidence"] > 0.5]

    for category in activated_categories:
        if category == "violence":
            results[category] = await run_violence_module(media)
        
        elif category == "traffic_accident":
            results[category] = await run_traffic_module(media)
        
        elif category == "weapons":
            results[category] = run_weapons_module(media)
        
        elif category == "fire":
            results[category] =  run_fire_smoke_module(media)
        
        elif category == "news":
            results[category] = await run_news_module(media)

        elif category == "normal":
            results[category] = {"message": "No specialized analysis needed for normal content."}
    print(f"Specialized module results:\n{results}\n")
    await vlm_manager.release()
    return results
