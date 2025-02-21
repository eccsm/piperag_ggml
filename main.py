from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from config import Config
from qa_service import QAChainBuilder, CatPersonality

app = FastAPI()

# Enable CORS.
app.add_middleware(
    CORSMiddleware,  # type: ignore
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instantiate configuration and services.
config = Config()
qa_service = QAChainBuilder(config)
cat_personality_service = CatPersonality()

@app.get("/ask")
async def ask(q: str = Query(..., description="User query")):
    """
    GET /ask?q=Your+question
    - If the query includes any RAG_KEYWORDS, use full RAG mode.
    - Otherwise, use a free chat mode with a cat personality.
    """
    if not q.strip():
        raise HTTPException(status_code=400, detail="Query is empty.")

    query_lower = q.lower()

    if any(keyword in query_lower for keyword in config.RAG_KEYWORDS):
        result = qa_service.chain({"query": q})
        answer = result["result"]
        return answer

    else:
        cat_name = cat_personality_service.get_personality()
        free_prompt = config.FREE_PROMPT_TEMPLATE.format(cat_name=cat_name, query=q)

        result = qa_service.llm(free_prompt, max_tokens=256)
        if isinstance(result, dict) and "choices" in result:
            full_answer = result["choices"][0]["text"].strip()
        else:
            full_answer = str(result).strip()

        # Clean up echoes from the prompt.
        if full_answer.startswith(free_prompt):
            full_answer = full_answer[len(free_prompt):].strip()
        if "User:" in full_answer:
            full_answer = full_answer.split("User:")[0].strip()

        return full_answer

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
