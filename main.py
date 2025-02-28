import asyncio
import logging
import os
from typing import Union, Dict, Any
from fastapi import FastAPI, HTTPException, Query, Body, File, UploadFile, BackgroundTasks, Depends, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security.api_key import APIKeyHeader, APIKey
import time
from contextlib import asynccontextmanager

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from services.image_recognition_service import ImageRecognitionService
from services.service_factory import ServiceFactory
from services.chat_service import ChatService
from config import Config

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("piperag")

# Define API key security scheme
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# Initialize config to get API key from .env through Pydantic
app_config = Config()

# Security dependency
async def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header == app_config.API_KEY:
        return api_key_header
    logger.warning(f"Invalid API key attempt")
    raise HTTPException(
        status_code=403, 
        detail="Invalid API Key"
    )

# Application state
class AppState:
    def __init__(self):
        self.config = Config()
        self.chat_service = None
        
    def initialize(self):
        self.chat_service = ServiceFactory.create_service(self.config)

# Initialize state on startup and cleanup on shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: initialize services
    app.state.app_state = AppState()
    app.state.app_state.initialize()
    logger.info("Application started, services initialized")
    yield
    # Shutdown: cleanup resources
    logger.info("Application shutting down, cleaning up resources")

# Create FastAPI app with lifespan manager
app = FastAPI(
    title="Piperag API",
    description="An API for chat and image recognition services.",
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS with restricted settings for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://resume-63067.web.app",
        "https://resume-63067.firebaseapp.com",
        "http://localhost:3000",  # For local development
        "http://localhost:5000"   # For local Firebase emulator
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],  # Include X-API-Key
    expose_headers=["X-Process-Time", "X-Error-Type"],
)

# Middleware for request logging and timing
@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = f"{time.time()}-{id(request)}"
    logger.info(f"Request started: {request.method} {request.url.path} (ID: {request_id})")
    start_time = time.time()
    
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        logger.info(f"Request completed: {request.method} {request.url.path} "
                   f"(ID: {request_id}) - Status: {response.status_code} - Time: {process_time:.4f}s")
        response.headers["X-Process-Time"] = f"{process_time:.4f}"
        return response
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(f"Request failed: {request.method} {request.url.path} "
                    f"(ID: {request_id}) - Error: {str(e)} - Time: {process_time:.4f}s")
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "error_type": type(e).__name__}
        )

# Dependency for getting chat service
def get_chat_service() -> ChatService:
    return app.state.app_state.chat_service

# Dependency for getting config
def get_config() -> Config:
    return app.state.app_state.config

# Define common error responses for API documentation
responses = {
    400: {"description": "Bad Request", "model": Dict[str, str]},
    403: {"description": "Forbidden - Invalid API Key", "model": Dict[str, str]},
    500: {"description": "Internal Server Error", "model": Dict[str, str]},
}

@app.get("/ask", summary="Chat Endpoint", description="Submit a query to the chat service and receive a response.", responses=responses, dependencies=[Depends(get_api_key)])
async def ask(
    q: str = Query(..., description="User query", min_length=1),
    model: str = Query(None, description="Optional model identifier"),
    chat_service: ChatService = Depends(get_chat_service),
    config: Config = Depends(get_config)
):
    try:
        # Switch model if needed
        if model and model != config.MODEL_TYPE:
            logger.info(f"Switching to {model}")
            config.MODEL_TYPE = model
            app.state.app_state.chat_service = ServiceFactory.create_service(config)
            chat_service = app.state.app_state.chat_service

        # Run the potentially blocking operation in a thread pool
        result = await asyncio.to_thread(chat_service.generate_response, q)
        return {"result": result}
    except ValueError as e:
        logger.warning(f"Invalid input: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Error getting response", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to generate response: {str(e)}",
            headers={"X-Error-Type": type(e).__name__}
        )

@app.post("/update_model", summary="Update Model", description="Update the backend model configuration.", responses=responses, dependencies=[Depends(get_api_key)])
async def update_model(
    background_tasks: BackgroundTasks,
    new_model: str = Body("", embed=True, description="Path to the new model file"),
    new_model_type: Union[str, None] = Body(None, embed=True, description="New model type (e.g., 'vicuna_ggml' or 'mlc_llm')"),
    config: Config = Depends(get_config)
):
    try:
        if new_model_type:
            if new_model_type not in ["vicuna_ggml", "mlc_llm"]:
                raise ValueError(f"Unsupported model type: {new_model_type}. Must be 'vicuna_ggml' or 'mlc_llm'")
            config.MODEL_TYPE = new_model_type

        if config.MODEL_TYPE == "vicuna_ggml":
            if new_model.strip():
                config.MODEL_PATH = new_model
        elif config.MODEL_TYPE == "mlc_llm":
            if new_model.strip():
                config.MLC_MODEL = new_model

        # Update the service in the application state
        app.state.app_state.chat_service = ServiceFactory.create_service(config)

        return {
            "status": "success", 
            "message": "Model updated successfully", 
            "model_type": config.MODEL_TYPE
        }
    except ValueError as e:
        logger.warning(f"Invalid model configuration: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Error updating model", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to update model: {str(e)}",
            headers={"X-Error-Type": type(e).__name__}
        )

@app.post("/recognize", summary="Image Recognition", description="Perform image recognition using the specified task.", responses=responses, dependencies=[Depends(get_api_key)])
async def recognize(
    task: str = Query(..., description="Task for image recognition"),
    file: UploadFile = File(..., description="Image file to analyze"),
    config: Config = Depends(get_config)
):
    # Validate the task parameter
    valid_tasks = list(config.IMAGE_MODEL_MAP.keys())
    if task not in valid_tasks:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid task: {task}. Must be one of {valid_tasks}"
        )
    
    # Check file type
    content_type = file.content_type
    if not content_type or not content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail=f"File must be an image. Got content type: {content_type}"
        )
        
    try:
        contents = await file.read()
        # Create a new instance of ImageRecognitionService with the task parameter
        image_service = ImageRecognitionService(config, task=task)
        result = await asyncio.to_thread(image_service.recognize, contents)
        return {"result": result}
    except ValueError as e:
        logger.warning(f"Invalid input for recognition: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to process image: {str(e)}",
            headers={"X-Error-Type": type(e).__name__}
        )

@app.get("/health", summary="Health Check", description="Check if the service is running properly")
async def health_check():
    return {"status": "healthy", "version": "1.0.0"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)