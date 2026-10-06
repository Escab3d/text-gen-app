from pydantic import BaseModel, Field

class GenerateRequest(BaseModel):
    """Input payload for text generation."""
    prompt: str = Field(default="", description="Optional input prompt for the model")
    max_new_tokens: int = Field(default=50, ge=1, le=512, description="Maximum number of tokens to generate")
    temperature: float = Field(default=0.7, ge=0.01, le=2.0, description="Sampling randomness (lower = more deterministic)")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Nucleus sampling threshold")
    top_k: int = Field(default=50, ge=0, description="Top-k filtering threshold")
    repetition_penalty: float = Field(default=1.2, ge=1.0, le=3.0, description="Penalty for repeating tokens")

class GenerateResponse(BaseModel):
    """Output payload from text generation."""
    prompt: str
    generated_text: str
    tokens_generated: int
    latency_seconds: float

class HealthResponse(BaseModel):
    """Service health and model status."""
    status: str
    model_loaded: bool
    device: str
