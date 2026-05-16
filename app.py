import numpy as np
import json
import nltk

from pathlib import Path
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
from google.colab import ai

# ----------------------------
# NLTK
# ----------------------------

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

transformer_model = SentenceTransformer(
    EMBEDDING_MODEL
)

# ----------------------------
# MEMORY
# ----------------------------

def load_memory(memory_file):

    memory_path = Path(memory_file)

    if memory_path.is_file():

        with open(memory_file, "r") as f:

            return json.load(f)

    return {
        "chat_history": [],
        "memory_summary": "",
        "facts": {
            "preferences": [],
            "user_info": []
        }
    }

# ----------------------------
# PDF EXTRACTION
# ----------------------------

def extract_text_from_pdf(pdf_file):

    reader = PdfReader(pdf_file)

    pages_data = []

    for page_number, page in enumerate(reader.pages,start=1):

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

def make_chunks(pages_data,chunk_size=CHUNK_SIZE,overlap_size=OVERLAP_SIZE):

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

            chunk_metadata = {

                "chunk_id": len(chunks),

                "page_number": page_number,

                "start_word": global_word_index,

                "word_count": word_count,

                "sentence_count": len(
                    chunk_sentences
                ),

                "text": chunk_text
            }

            chunks.append(chunk_metadata)

            global_word_index += word_count

            start += (
                chunk_size - overlap_size
            )

    return chunks

# ----------------------------
# EMBEDDINGS
# ----------------------------

def create_embeddings(chunks):

    chunk_texts = [

        chunk["text"]

        for chunk in chunks
    ]

    embeddings = transformer_model.encode(

        chunk_texts,

        normalize_embeddings=True
    )

    return np.array(
        embeddings,
        dtype=np.float32
    )

# ----------------------------
# QUERY EXPANSION
# ----------------------------

def expand_query(query):

    query_lower = query.lower()

    if "theme" in query_lower:

        return (
            query
            + " main idea moral lesson"
        )

    if "values" in query_lower:

        return (
            query
            + " ethics teachings morals"
        )

    if "compare" in query_lower:

        return (
            query
            + " differences similarities"
        )

    return query

# ----------------------------
# COSINE SEARCH
# ----------------------------

def cosine_search(
    query_vector,
    chunk_embeddings
):

    similarity_scores = np.dot(
        chunk_embeddings,
        query_vector
    )

    ranked_scores = []

    for index, score in enumerate(
        similarity_scores
    ):

        ranked_scores.append(
            (score, index)
        )

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

def retrieve_chunks(
    user_query,
    chunks,
    chunk_embeddings
):

    expanded_query = expand_query(
        user_query
    )

    query_embedding = transformer_model.encode(

        [expanded_query],

        normalize_embeddings=True
    )

    query_vector = query_embedding[0]

    top_results = cosine_search(

        query_vector,

        chunk_embeddings
    )

    retrieved_results = []

    for score, index in top_results:

        retrieved_results.append({

            "score": float(score),

            "chunk": chunks[index]
        })

    return retrieved_results

# ----------------------------
# CONTEXT BUILDER
# ----------------------------

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
# ANSWER GENERATION
# ----------------------------

def generate_answer(prompt):

    response = ai.generate_text(prompt)

    answer_text = (

        response.text

        if hasattr(response, "text")

        else str(response)
    )

    return answer_text

# ----------------------------
# BUILD PIPELINE
# ----------------------------

pages_data = extract_text_from_pdf(
    PDF_FILE
)

chunks = make_chunks(
    pages_data
)

chunk_embeddings = create_embeddings(
    chunks
)

print("✅ RAG System Ready")

# ----------------------------
# CHAT LOOP
# ----------------------------

while True:

    user_query = input(
        "\nAsk your question: "
    )

    if user_query.lower() in [

        "exit",
        "quit",
        "stop",
        "break"

    ]:

        print("Exiting...")
        break

    memory_data_dict = load_memory(
        MEMORY_FILE
    )

    retrieved_results = retrieve_chunks(

        user_query,

        chunks,

        chunk_embeddings
    )

    print("\nRetrieved Chunks:\n")

    for rank, item in enumerate(

        retrieved_results,

        start=1
    ):

        score = item["score"]

        chunk = item["chunk"]

        print(f"Rank: {rank}")

        print(
            f"Score: {score:.4f}"
        )

        print(
            f"Chunk ID: {chunk['chunk_id']}"
        )

        print(
            f"Page: {chunk['page_number']}"
        )

        print(
            f"Start Word: {chunk['start_word']}"
        )

        print(
            f"Sentence Count: {chunk['sentence_count']}"
        )

        print(
            f"Word Count: {chunk['word_count']}"
        )

        print("\nChunk Preview:\n")

        print(chunk["text"][:500])

        print("\n" + "-" * 60)

    best_score = retrieved_results[0]["score"]

    if best_score < MIN_SIMILARITY_THRESHOLD:

        print(
            "\nEvidence is weak or missing."
        )

        continue

    context = build_context(
        retrieved_results
    )

    recent_chat = ""

    for question, answer in memory_data_dict[
        "chat_history"
    ][-3:]:

        recent_chat += f"""

Q: {question}

A: {answer}

"""

    prompt = f"""
You are a precise AI document assistant.

Rules:
1. Use ONLY retrieved context
2. No hallucination
3. If unsure say:
   Evidence is weak or missing
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

    answer = generate_answer(
        prompt
    )

    print("\nAnswer:\n")

    print(answer)

    # ----------------------------
    # MEMORY SAVE
    # ----------------------------

    memory_data_dict[
        "chat_history"
    ].append(

        (user_query, answer)
    )

    with open(
        MEMORY_FILE,
        "w"
    ) as f:

        json.dump(
            memory_data_dict,
            f,
            indent=2
        )

    # ----------------------------
    # FACT STORAGE
    # ----------------------------

    query_lower = user_query.lower()

    if (

        "favorite" in query_lower

        or "i like" in query_lower

    ):

        if user_query not in memory_data_dict[
            "facts"
        ]["preferences"]:

            memory_data_dict[
                "facts"
            ]["preferences"].append(
                user_query
            )

    if (

        "my name is" in query_lower

        or "i am" in query_lower

    ):

        if user_query not in memory_data_dict[
            "facts"
        ]["user_info"]:

            memory_data_dict[
                "facts"
            ]["user_info"].append(
                user_query
            )

    with open(
        MEMORY_FILE,
        "w"
    ) as f:

        json.dump(
            memory_data_dict,
            f,
            indent=2
        )
