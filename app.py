from fastapi import FastAPI
from pydantic import BaseModel
from rag import ask_portfolio
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# 🔥 Add this
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # in production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    question: str

@app.post("/chat")
def chat(query: Query):
    answer = ask_portfolio(query.question)
    return {"answer": answer}

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)