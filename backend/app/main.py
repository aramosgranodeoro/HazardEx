import base64

from fastapi import FastAPI, File, UploadFile
from app.triage.triage import classify_image, run_specialized_modules

app = FastAPI(
    title="HazardEx",
    version="1.0.0"
)

@app.get("/")
def root():
    return {"message": "Backend funcionando"}

@app.get("/health")
def health():
    return {"status": "ok"}

# Crear conversación y almacenar el estado inicial en memoria

# Borrar conversación y estado asociado al thread_id

# Análisis inicial del medio (imagen o vídeo) y generación de contexto para el agente
@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    media_bytes = await file.read()
    categories, image = classify_image(media_bytes, file.filename)
#     categories = """
# {
#                     "predicted_categories": [
#                         {
#                             "category": "fire",
#                             "confidence": 0.00
#                         },
#                         {
#                             "category": "normal",
#                             "confidence": 0.15
#                         },
#                         {
#                             "category": "violence",
#                             "confidence": 0.98
#                         },
#                         {
#                             "category": "weapons",
#                             "confidence": 0.01
#                         },
#                         {
#                             "category": "traffic_accident",
#                             "confidence": 0.01
#                         },
#                         {
#                             "category": "news",
#                             "confidence": 0.00
#                         }
#                     ],
#                     "description": "The image prominently features a large bonfire with intense flames and smoke rising near a body of water, clearly fitting the fire category."
#                 }
# """
#     image = base64.b64encode(media_bytes).decode('utf-8')
    result = await run_specialized_modules(categories, image)
        
    # thread_id = str(uuid.uuid4())
    # initial_state = {
    #     "messages": [SystemMessage(content=format_analysis_as_context(result))],
    #     "vlm_context": result,
    # }
    # checkpointer.put(config={"configurable": {"thread_id": thread_id}}, state=initial_state)
    
    # return {"thread_id": thread_id, "analysis": result}
    return {"message": "Análisis simulado, funcionalidad en desarrollo."}

@app.post("/query")
async def query(thread_id: str, question: str):
    # state = checkpointer.get(thread_id)
    # if not state:
    #     return {"error": "Thread ID no encontrado"}
    
    # messages = state["messages"] + [HumanMessage(content=question)]
    # new_state = agent.invoke({"messages": messages})
    
    # checkpointer.put(config={"configurable": {"thread_id": thread_id}}, state=new_state)
    
    # return {"response": new_state["messages"][-1].content}
    return {"message": "Consulta simulada, funcionalidad en desarrollo."}