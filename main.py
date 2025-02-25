import asyncio
import logging
import os
from typing import Union

from fastapi import FastAPI, HTTPException, Query, Body, File, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware



os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from services.image_recognition_service import ImageRecognitionService
from services.service_factory import ServiceFactory
from config import Config
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("piperag")

app = FastAPI(
    title="Piperag API",
    description="An API for chat and image recognition services.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

config = Config()

# Use the ServiceFactory instead of the custom function
chat_service = ServiceFactory.create_service(config)
current_model_type = config.MODEL_TYPE  # track the current model type globally

@app.get("/ask", summary="Chat Endpoint", description="Submit a query to the chat service and receive a response.")
async def ask(
    q: str = Query(..., description="User query"),
    model: str = Query(None, description="Optional model identifier")
):
    if not q.strip():
        raise HTTPException(status_code=400, detail="Query is empty.")

    try:
        global current_model_type, chat_service
        # Switch model if needed
        if model and model != current_model_type:
            logger.info(f"Switching to {model}")
            config.MODEL_TYPE = model
            current_model_type = model
            chat_service = ServiceFactory.create_service(config)

        # Run the potentially blocking operation in a thread pool
        result = await asyncio.to_thread(chat_service.generate_response, q)
        return {"result": result}
    except Exception as e:
        logger.error("Error getting response", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/update_model", summary="Update Model", description="Update the backend model configuration.")
async def update_model(
    background_tasks: BackgroundTasks,
    new_model: str = Body("", embed=True, description="Path to the new model file"),
    new_model_type: Union[str, None] = Body(None, embed=True, description="New model type (e.g., 'vicuna_ggml' or 'mlc_llm')")
):
    try:
        if new_model_type:
            config.MODEL_TYPE = new_model_type

        if config.MODEL_TYPE == "vicuna_ggml":
            if new_model.strip():
                config.MODEL_PATH = new_model
        elif config.MODEL_TYPE == "mlc_llm":
            if new_model.strip():
                config.DOMAIN_MODEL_PATH = new_model

        global chat_service
        chat_service = ServiceFactory.create_service(config)


        return {"status": "Model reload started", "model_type": config.MODEL_TYPE}
    except Exception as e:
        logger.error("Error updating model", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/recognize", summary="Image Recognition", description="Perform image recognition using the specified task.")
async def recognize(
    task: str = Query(..., description="Task for image recognition"),
    file: UploadFile = File(..., alias="image")
):
    try:
        contents = await file.read()
        # Create a new instance of ImageRecognitionService with the task parameter
        image_service = ImageRecognitionService(config, task=task)
        result = await asyncio.to_thread(image_service.recognize, contents)
        return {"result": result}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
