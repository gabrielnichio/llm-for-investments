from assistant import Assistant
from tools.document_display import display_similar_documents, display_source_documents
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from pydantic import BaseModel

class Infos(BaseModel):
    question: str

app = FastAPI(
    title="API Assistente de Investimentos",
    description="API para análise de investimentos com IA",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Em produção, substitua por origens específicas
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

assistant = Assistant()
llm = assistant.llm()

@app.get("/")
async def root():
    return {"message": "API online"}

@app.post("/investments/")
async def analisys(front_infos: Infos):
    query_embeddings = assistant.get_embedding().get_text_embedding(front_infos.question)
    chroma_collection = assistant.get_chroma_collection()
    
    display_similar_documents(chroma_collection, query_embeddings)

    response = llm.chat(front_infos.question)

    display_source_documents(response)

    return {"response": response.response}




if __name__ == "__main__":
    print("Initializing backend")
    uvicorn.run(app, host="0.0.0.0", port=8000)
