from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from app.schemas import GenerateRequest, GenerateResponse, HealthResponse
from app.model_loader import model_service

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event: loads model once when the server boots up."""
    print("[STARTUP] Initializing model service...")
    model_service.load()
    yield
    print("[SHUTDOWN] Cleaning up resources...")

app = FastAPI(
    title="Text Generation Model API",
    description="A FastAPI backend serving a fine-tuned text generation model",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health_check():
    """Health check endpoint to monitor API and model readiness."""
    return HealthResponse(
        status="healthy" if model_service.is_loaded else "model_not_ready",
        model_loaded=model_service.is_loaded,
        device=model_service.device
    )

@app.post("/generate", response_model=GenerateResponse, tags=["Inference"])
def generate_text(request: GenerateRequest):
    """Generate continuation text based on the provided prompt and parameters."""
    if not model_service.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model is not ready. Verify that weights exist in the model directory."
        )

    try:
        result = model_service.generate(
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            repetition_penalty=request.repetition_penalty
        )
        return GenerateResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
