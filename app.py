import streamlit as st
import pandas as pd
import faiss
import pickle
import os
import psycopg2 # connect to data server - postgresDB - customer , payment 

from sentence_transformers import SentenceTransformer
from transformers import pipeline
from huggingface_hub import login

# -------------------------------
# Step 1: Load dataset (CSV + SQL)
# -------------------------------
# Movies CSV
df_movies = pd.read_csv("movies.csv")  # columns: title, description, genre

# PostgreSQL connection
conn = psycopg2.connect(
    host="localhost",
    database="moviesdb",
    user="postgres",
    password="harsha",
    port="5432"
)

# Customers table
df_customers = pd.read_sql("SELECT * FROM customer;", conn)

# Payments table
df_payments = pd.read_sql("SELECT * FROM payment;", conn)

conn.close()

# -------------------------------
# Step 2: Prepare combined corpus
# -------------------------------
# Normalize into text documents
movie_docs = [f"Movie: {row['title']} ({row['genre']}) - {row['description']}" 
              for _, row in df_movies.iterrows()]

customer_docs = [f"Customer: {row['first_name']} (ID: {row['customer_id']}, Email: {row['email']})"
                 for _, row in df_customers.iterrows()]

payment_docs = [f"Payment: CustomerID {row['customer_id']} paid {row['amount']} on {row['payment_date']}"
                for _, row in df_payments.iterrows()]

all_docs = movie_docs + customer_docs + payment_docs

# -------------------------------
# Step 3: Embeddings with caching
# -------------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")

if os.path.exists("all_embeddings.pkl"):
    with open("all_embeddings.pkl", "rb") as f:
        all_embeddings = pickle.load(f)
else:
    all_embeddings = embedder.encode(all_docs, convert_to_numpy=True)
    with open("all_embeddings.pkl", "wb") as f:
        pickle.dump(all_embeddings, f)

# -------------------------------
# Step 4: Build FAISS index
# -------------------------------
embedding_dim = all_embeddings.shape[1]
index = faiss.IndexFlatL2(embedding_dim)
index.add(all_embeddings)

# -------------------------------
# Step 5: Load LLM (Hugging Face)
# -------------------------------

HF_TOKEN = ""
login(HF_TOKEN)

qa_pipeline = pipeline(
    "text-generation",
    model="tiiuae/falcon-7b-instruct",
    max_new_tokens=200
)

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("📚 Unified RAG Chatbot (Movies + Customers + Payments)")

query = st.text_input("Ask me about movies, customers, or payments:")
genre_filter = st.selectbox("Filter by genre (movies only):", ["All"] + df_movies["genre"].unique().tolist())

if query:
    # -------------------------------
    # Step 6: Apply genre filter (movies only)
    # -------------------------------
    filtered_docs = all_docs
    filtered_embeddings = all_embeddings

    if genre_filter != "All":
        movie_subset = [f"Movie: {row['title']} ({row['genre']}) - {row['description']}" 
                        for _, row in df_movies[df_movies["genre"] == genre_filter].iterrows()]
        filtered_docs = movie_subset + customer_docs + payment_docs
        filtered_embeddings = embedder.encode(filtered_docs, convert_to_numpy=True)

        index = faiss.IndexFlatL2(filtered_embeddings.shape[1])
        index.add(filtered_embeddings)
    else:
        index = faiss.IndexFlatL2(filtered_embeddings.shape[1])
        index.add(filtered_embeddings)

    # -------------------------------
    # Step 7: Retrieval
    # -------------------------------
    query_embedding = embedder.encode([query])
    D, I = index.search(query_embedding, k=10)  # top 10 results
    retrieved = [filtered_docs[i] for i in I[0]]

    # -------------------------------
    # Step 8: Build context
    # -------------------------------
    context = "\n".join(retrieved)

    # -------------------------------
    # Step 9: Generation
    # -------------------------------
    prompt = f"Answer the question based on these records:\n{context}\n\nQuestion: {query}\nAnswer:"
    answer = qa_pipeline(prompt)[0]["generated_text"]

    # -------------------------------
    # Step 10: Display results
    # -------------------------------
    st.subheader("📂 Retrieved Records")
    for doc in retrieved:
        st.write(doc)

    st.subheader("🤖 Chatbot Answer")
    st.markdown(answer)
