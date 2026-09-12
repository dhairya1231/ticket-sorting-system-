from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "customer_tweets"

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")

# Reopen the store you already built — don't re-embed, just load it
vector_store = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embeddings,
    collection_name=COLLECTION_NAME,
)

def retrieve_similar_tweets(query: str, k: int = 4):
    """
    Retrieve the k customer tweets most similar to `query`.
    Returns a list of dicts: [{text, tweet_id, author_id, score}, ...]
    """
    results = vector_store.similarity_search_with_relevance_scores(query, k=k)

    matches = []
    for doc, score in results:
        matches.append({
            "text": doc.page_content,
            "tweet_id": doc.metadata.get("tweet_id"),
            "author_id": doc.metadata.get("author_id"),
            "score": score,
        })
    return matches


if __name__ == "__main__":
    test_query = "my package never arrived and tracking says delivered"
    matches = retrieve_similar_tweets(test_query, k=4)

    print(f"Query: {test_query}\n")
    for i, m in enumerate(matches, start=1):
        print(f"[{i}] score={m['score']:.3f}  tweet_id={m['tweet_id']}  author={m['author_id']}")
        print(f"    {m['text']}\n")