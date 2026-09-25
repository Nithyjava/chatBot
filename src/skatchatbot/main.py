from dotenv import load_dotenv

from fastapi import (
    FastAPI,
    WebSocket,
    WebSocketDisconnect,
    HTTPException,
)

from fastapi.middleware.cors import CORSMiddleware

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

from skatchatbot.schema import MongoDbSchema
from skatchatbot.config import docs_collection

load_dotenv()

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.3,
)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

@app.get("/health-check")
def health_check():
    return {
        "status": True,
        "message": "server is running",
    }

@app.get("/chat")
def chat_bot(query: str):

    try:
        response = llm.invoke(query)

        return {
            "status": True,
            "query": query,
            "response": response.content,
        }
    except Exception as e:
        return {
            "status": False,
            "error": str(e),
        }

@app.post("/creat-test")
def create_test(user: MongoDbSchema):
    try:
        result = docs_collection.insert_one(
            user.model_dump()
        )
        return {
            "status": True,
            "message": "User inserted successfully",
            "inserted_id": str(result.inserted_id),
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )

@app.post("/chunk")
def document_chunking():
    try:
        loader = PyPDFLoader(
            "./skattech_overview_kb.pdf"
        )
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )

        chunks = text_splitter.split_documents(docs)
        texts = [
            chunk.page_content
            for chunk in chunks
        ]
        vectors = embeddings.embed_documents(texts
        )

        documents = []

        for index, (chunk, vector) in enumerate(
            zip(chunks, vectors)
        ):

            documents.append({

                "chunk_index": index,

                "text": chunk.page_content,

                "embedding": vector,

                "metadata": chunk.metadata,

            })


        result = docs_collection.insert_many(
            documents
        )


        return {

            "status": True,

            "message": (
                "PDF chunked and stored successfully"
            ),

            "total_pages": len(docs),

            "total_chunks": len(chunks),

            "inserted_count": len(
                result.inserted_ids
            ),

        }

    except Exception as e:

        print(
            "Chunking error:",
            str(e)
        )

        return {

            "status": False,

            "error": str(e),

        }

def bot_chat(query: str):

    try:
        query_embedding = (
            embeddings.embed_query(query)
        )
        results = docs_collection.aggregate([

            {

                "$vectorSearch": {

                    "index": "default",

                    "path": "embedding",

                    "queryVector": query_embedding,

                    "numCandidates": 100,

                    "limit": 5,

                }

            },

            {

                "$project": {

                    "_id": 0,

                    "text": 1,

                    "score": {
                        "$meta": "vectorSearchScore"
                    },

                }

            },

        ])

        documents = list(results)
        context_text = "\n\n".join(

            doc["text"]

            for doc in documents

        )

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are Jarvis, the official AI assistant for SKATTECH. "
                "You represent SKATTECH and help users with questions about "
                "SKATTECH, its products, services, careers, jobs, support, "
                "company information, and other information available in your "
                "knowledge base. "

                "Use the document below as your primary source of truth. "
                "Only provide specific factual information when it is available "
                "in the document. Never invent, assume, or fabricate SKATTECH "
                "company information. "

                "You are a conversational company assistant, not a generic AI "
                "assistant. Keep your responses relevant to SKATTECH whenever "
                "the question is related to the company. "

                "If the user asks about information that is not available in "
                "the document, do not give unrelated external recommendations "
                "such as visiting another company's website or contacting an "
                "email address unless that information is explicitly available "
                "in the document. "

                "Instead, clearly tell the user that the current information "
                "is not available in your SKATTECH knowledge base. "
                "Then offer a helpful next step based on what you can actually "
                "do. "

                "For example, if the user asks about current SKATTECH hiring "
                "information and no hiring information exists in the document, "
                "respond naturally like: "
                "\"I don't have the current SKATTECH hiring information available "
                "right now. I can help you with the SKATTECH information available "
                "in my knowledge base.\" "

                "Do not pretend that information exists when it does not. "

                "If the question is outside the available SKATTECH knowledge, "
                "be honest about the limitation instead of guessing. "

                "Reply in plain, natural conversational text only. "
                "Do not use Markdown formatting. "
                "Do not use **, ##, tables, bullet symbols, or horizontal rules. "
                "Use plain hyphens (-) when needed. "
                "Use straight quotes only. "
                "Never use smart quotes or typographic dashes. "

                "Keep answers concise, friendly, and professional. "
                "Speak like an official SKATTECH assistant. "

                "\n\n"
                "SKATTECH Knowledge Base:\n"
                "{documents}"
            ),
            (
                "human",
                "{input}"
            ),
        ])

        chain = prompt | llm

        response = chain.invoke({
            "documents": context_text,
            "input": query,
        })
        return response.content
    except Exception as e:
        raise



class ConnectionManager:

    def __init__(self):

        self.connections: list[WebSocket] = []


    async def connect(
        self,
        websocket: WebSocket,
    ):

        await websocket.accept()

        self.connections.append(
            websocket
        )

        print(
            "WebSocket connected"
        )


    def disconnect(
        self,
        websocket: WebSocket,
    ):

        if websocket in self.connections:

            self.connections.remove(
                websocket
            )

        print(
            "WebSocket disconnected"
        )


    async def broadcast(
        self,
        data: dict,
    ):

        for connection in self.connections:

            await connection.send_json(
                data
            )

manager = ConnectionManager()

@app.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
):

    await manager.connect(
        websocket
    )

    try:
        while True:
            data = await websocket.receive_json()
            print("Received from frontend:")
            username = data.get(
                "username",
                "User",
            )

            message = data.get(
                "message",
                "",
            )
            if not message:
                await websocket.send_json({
                    "username": "Jarvis",
                    "message": (
                        "Please enter a message."
                    ),
                })

                continue

            response = bot_chat(
                message
            )

            await websocket.send_json({
                "username": "Jarvis",
                "message": response,
            })

            print(
                "Response sent to frontend"
            )

    except WebSocketDisconnect:

        manager.disconnect(
            websocket
        )

        print("Client disconnected")
    except Exception as e:
        manager.disconnect(websocket)
        print(
            "WebSocket error:",
            str(e)
        )