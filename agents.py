import pandas as pd
import requests
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_chroma import Chroma
load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")
CSV_PATH = "sample.csv"
df = pd.read_csv(CSV_PATH)
customer_tweets = df[df["inbound"] == True].copy()
brand_replies = df[df["inbound"] == False].copy()
print(f"Loaded {len(customer_tweets)} customer tweets")
print(f"Loaded {len(brand_replies)} brand replies")


docs = [
    Document(
        page_content=row["text"],
        metadata={
            "tweet_id": int(row["tweet_id"]),
            "author_id": str(row.get("author_id", "")),
        }
    )
    for _, row in customer_tweets.iterrows()
]

print(f"Prepared {len(docs)} documents for embedding")
vector_store = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    collection_name="customer_tweets",
    persist_directory="./chroma_db",
)

print("Stored in Chroma at ./chroma_db")