import logging
import os
from typing import Union

from fastapi import FastAPI, HTTPException, Query, Body, File, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

from chat_service import ChatService
from config import Config

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("piperag")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

config = Config()


# Factory function to select the appropriate chat service.
def get_chat_service() -> ChatService:
    if config.MODEL_TYPE == "vicuna_ggml":
        from rag_chat_service import RAGChatService
        return RAGChatService(config)
    elif config.MODEL_TYPE == "mlc_llm":
        from basic_chat_service import BasicChatService
        return BasicChatService(config)
    else:
        raise ValueError("Unsupported MODEL_TYPE in config.")


chat_service = get_chat_service()
current_model_type = config.MODEL_TYPE  # track the current model type globally


@app.get("/ask")
async def ask(
    q: str = Query(..., description="User query"),
    model: str = Query(None, description="Optional model identifier")
):
    if not q.strip():
        raise HTTPException(status_code=400, detail="Query is empty.")
    try:
        global current_model_type, chat_service
        # If the query specifies a model and it differs from the current one,
        # reinitialize the chat service with the correct configuration.
        if model and model != current_model_type:
            if model == "mlc_llm":
                logger.info("Switching to mlc_llm")
            else:
                logger.info("Switching to vicuna_ggml")

            config.MODEL_TYPE = model
            current_model_type = model
            chat_service = get_chat_service()
        result = chat_service.get_response(q)
        return {"result": result}
    except Exception as e:
        logger.error("Error getting response", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/update_model")
async def update_model(
        background_tasks: BackgroundTasks,
        new_model: str = Body("", embed=True, description="Path to the new model file"),
        new_model_type: Union[str, None] = Body(None, embed=True,
                                                description="New model type (e.g., 'vicuna_ggml' or 'mlc_llm')")
):
    try:
        def reload_model_task():
            if new_model_type:
                config.MODEL_TYPE = new_model_type

            if config.MODEL_TYPE == "vicuna_ggml":
                if new_model.strip():
                    config.MODEL_PATH = new_model
            elif config.MODEL_TYPE == "mlc_llm":
                if new_model.strip():
                    config.DOMAIN_MODEL_PATH = new_model

            global chat_service

            chat_service = get_chat_service()

        background_tasks.add_task(reload_model_task)
        return {"status": "Model reload started", "model_type": config.MODEL_TYPE}
    except Exception as e:
        logger.error("Error updating model", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/recognize")
async def recognize(
    task: str = Query(..., description="Task for image recognition, e.g., 'deepfake_detection'"),
    file: UploadFile = File(...,alias="image")
):
    try:
        contents = await file.read()
        from image_recognition_service import ImageRecognitionService
        # Pass the task to the service so it can select the correct model.
        image_service = ImageRecognitionService(config, task)
        result = image_service.recognize(contents)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
