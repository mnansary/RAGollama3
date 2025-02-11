from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from typing import Generator
from model import BanglaRAGService  # Import your BanglaRAGService class

app = FastAPI()
rag_service = BanglaRAGService()

@app.get("/query/")
async def query(question: str):
    async def event_stream() -> Generator[str, None, None]:
        for chunk in rag_service.process_query(question):
            yield chunk

    return StreamingResponse(event_stream(), media_type="text/plain")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3030)
