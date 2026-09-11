vector_store = Chroma.from_documents(
#     documents=docs,
#     embedding=embeddings,
#     collection_name="customer_tweets",
#     persist_directory="./chroma_db",  # folder where it saves to disk
# )

# print("Stored in Chroma at ./chroma_db")