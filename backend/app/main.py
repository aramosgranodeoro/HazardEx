from fastapi import FastAPI

app = FastAPI(
    title="HazardEx",
    version="1.0.0"
)

@app.get("/")
def root():
    return {"message": "Backend funcionando"}