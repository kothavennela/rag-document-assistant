import numpy as np
import json
import nltk

from pathlib import Path
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
from google.colab import ai

# Download NLTK resources
nltk.download("punkt")
nltk.download("punkt_tab")

# ----------------------------
# CONFIG
# ----------------------------

PDF_FILE = "Grandma's Bag of Stories by Sudha Murthy.pdf"

MEMORY_FILE = "memory.json"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

MIN_SIMILARITY_THRESHOLD = 0.30

CHUNK_SIZE = 8
OVERLAP_SIZE = 2

# ----------------------------
# MODEL
# ----------------------------

transformer_model = SentenceTransformer(EMBEDDING_MODEL)

# ----------------------------
# MEMORY
# ----------------------------

# Load chat history from JSON
def load_memory(memory_file):

    memory_path = Path(memory_file)

    if memory_path.is_file():

        with open(memory_file, "r") as f:

            return json.load(f)

    return {
        "chat_history": []
    }


# Save chat history to JSON
def save_memory(memory_file, memory_data):

    with open(memory_file, "w") as f:

        json.dump(memory_data, f, indent=2)


# ----------------------------
# PDF EXTRACTION
# ----------------------------

# Extract text page by page from PDF
def extract_text_from_pdf(pdf_file):

    reader = PdfReader(pdf_file)

    pages_data = []

    for page_number, page in enumerate(reader.pages, start=1):

        extracted_text = page.extract_text()

        if extracted_text:

            pages_data.append({
                "page_number": page_number,
                "text": extracted_text
            })

    return pages_data


# ----------------------------
# CHUNKING
# ----------------------------

# Split PDF text into overlapping sentence chunks
def make_chunks(pages_data, chunk_size=CHUNK_SIZE, overlap_size=OVERLAP_SIZE):

    chunks = []

    global_word_index = 0

    for page_data in pages_data:

        page_number = page_data["page_number"]

        text = page_data["text"]

        sentences = sent_tokenize(text)

        start = 0

        while start < len(sentences):

            end = start + chunk_size

            chunk_sentences = sentences[start:end]

            chunk_text = " ".join(chunk_sentences)

            word_count = len(chunk_text.split())

            chunks.append({

                "chunk_id": len(chunks),

                "page_number": page_number,

                "start_word": global_word_index,

                "word_count": word_count,

                "sentence_count": len(chunk_sentences),

                "text": chunk_text
            })

            global_word_index += word_count

            start += chunk_size - overlap_size

    return chunks


# ----------------------------
# EMBEDDINGS
# ----------------------------

# Convert chunk sentences text into embeddings
def create_embeddings(chunks):

    chunk_texts = [chunk["text"] for chunk in chunks]

    embeddings = transformer_model.encode(chunk_texts,normalize_embeddings=True)

    return np.array(embeddings,dtype=np.float32)


# ----------------------------
# QUERY EXPANSION
# ----------------------------

# Add keywords to improve retrieval
def expand_query(query):

    query_lower = query.lower()

    if "theme" in query_lower:

        return query + " main idea moral lesson"

    if "values" in query_lower:

        return query + " ethics teachings morals"

    if "compare" in query_lower:

        return query + " differences similarities"

    return query


# ----------------------------
# COSINE SEARCH
# ----------------------------

# Find most relevant chunks using similarity search
# Find closest embedding to user query 
def cosine_search(query_vector, chunk_embeddings):

    similarity_scores = np.dot(chunk_embeddings,query_vector)

    ranked_scores = []

    for index, score in enumerate(similarity_scores):

        ranked_scores.append((float(score), index))

    ranked_scores = sorted(
        ranked_scores,
        key=lambda x: x[0],
        reverse=True
    )

    best_score = ranked_scores[0][0]

    if best_score > 0.65:
        top_k = 2

    elif best_score > 0.45:
        top_k = 4

    else:
        top_k = 6

    return ranked_scores[:top_k]


# ----------------------------
# RETRIEVAL
# ----------------------------

# Retrieve top matching chunks between user query and chunk embeddings from pdf
def retrieve_chunks(user_query, chunks, chunk_embeddings):

    expanded_query = expand_query(user_query)
    #converting user query text to embeddings
    query_embedding = transformer_model.encode(
        [expanded_query],
        normalize_embeddings=True
    )

    query_vector = query_embedding[0]

    top_results = cosine_search(query_vector,chunk_embeddings)

    retrieved_results = []

    for score, index in top_results:

        retrieved_results.append({

            "score": score,

            "chunk": chunks[index]
        })

    return retrieved_results


# ----------------------------
# CONTEXT BUILDER
# ----------------------------

# Convert retrieved chunks into prompt context
def build_context(retrieved_results):

    context = ""

    for item in retrieved_results:

        score = item["score"]

        chunk = item["chunk"]

        if score >= MIN_SIMILARITY_THRESHOLD:

            context += f"""

                Chunk ID: {chunk["chunk_id"]}

                Page Number: {chunk["page_number"]}

                Similarity Score: {score:.4f}

                Text:
                {chunk["text"]}

                """

    return context


# ----------------------------
# LLM
# ----------------------------

# Generate answer from LLM
def generate_answer(prompt):

    response = ai.generate_text(prompt)

    answer_text = (
        response.text
        if hasattr(response, "text")
        else str(response)
    )

    return answer_text


# ----------------------------
# INITIALIZATION
# ----------------------------

# Build chunks and embeddings once
def initialize_rag(PDF_FILE):

    pages_data = extract_text_from_pdf(PDF_FILE)

    chunks = make_chunks(pages_data)

    chunk_embeddings = create_embeddings(chunks)

    return chunks, chunk_embeddings


# ----------------------------
# MAIN RAG PIPELINE
# ----------------------------

# Run retrieval + prompting + answer generation
def ask_rag(user_query, chunks, chunk_embeddings):

    memory_data = load_memory(MEMORY_FILE)

    retrieved_results = retrieve_chunks(
        user_query,
        chunks,
        chunk_embeddings
    )

    best_score = retrieved_results[0]["score"]

    if best_score < MIN_SIMILARITY_THRESHOLD:

        return {
            "answer": "Evidence is weak or missing.",
            "retrieved_results": retrieved_results,
            "best_score": best_score
        }

    context = build_context(retrieved_results)

    recent_chat = ""

    for question, answer in memory_data["chat_history"][-3:]:

        recent_chat += f"""

                        Q: {question}

                        A: {answer}

                        """

    prompt = f"""
            You are a precise AI document assistant.

            Rules:
            1. Use ONLY retrieved context
            2. Do not hallucinate
            3. If evidence is weak, say so
            4. Be concise
            5. Mention uncertainty honestly

            Recent Conversation:
            {recent_chat}

            Retrieved Context:
            {context}

            Question:
            {user_query}

            Answer:
            """

    answer = generate_answer(prompt)

    memory_data["chat_history"].append((user_query, answer))

    save_memory(MEMORY_FILE, memory_data)

    return {
        "answer": answer,
        "retrieved_results": retrieved_results,
        "best_score": best_score
    }
