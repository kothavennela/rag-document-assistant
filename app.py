import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from pathlib import Path
import json
from google.colab import ai

# ----------------------------
# Setup
# ----------------------------

transformer_model = SentenceTransformer('all-MiniLM-L6-v2')

pdf_file = "Grandma's Bag of Stories by Sudha Murthy.pdf"
memory_file = "sample (1).json"

llm_model = "ai"

TOP_K = 3
MIN_SIMILARITY_THRESHOLD = 0.30

# ----------------------------
# Memory Loader
# ----------------------------

def load_memory(memory_file):

    memory_path = Path(memory_file)

    if memory_path.is_file():

        print("file exists")

        with open(memory_file, 'r') as f:
            memory_data_dict = json.load(f)

    else:

        memory_data_dict = {
            "chat_history": [],
            "memory_summary": "",
            "facts": {
                "preferences": [],
                "user_info": []
            }
        }

    return memory_data_dict

# ----------------------------
# PDF Text Extraction
# ----------------------------

def extract_text(pdf_file):

    reader = PdfReader(pdf_file)

    text = ""

    for page in reader.pages:
        extracted = page.extract_text()

        if extracted:
            text += extracted + "\n"

    return text

# ----------------------------
# Better Chunking
# ----------------------------

def make_chunks(text, chunk_size=500, overlap_size=50):

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunk_words = words[start:end]

        chunk_text = " ".join(chunk_words)

        chunks.append({
            "chunk_id": len(chunks),
            "text": chunk_text
        })

        start += chunk_size - overlap_size

    chunk_texts = [chunk["text"] for chunk in chunks]

    chunks_embeddings = transformer_model.encode(
        chunk_texts,
        normalize_embeddings=True
    )

    return chunks, chunks_embeddings

# ----------------------------
# Query Expansion
# ----------------------------

def expand_query(query):

    query_lower = query.lower()

    if "theme" in query_lower:
        return query + " main idea message moral lesson"

    if "central idea" in query_lower:
        return query + " main message lesson"

    if "values" in query_lower:
        return query + " morals ethics teachings"

    return query

# ----------------------------
# Cosine Similarity Search
# ----------------------------

def cosine_search(query_vector, chunks_embeddings):

    scores = np.dot(chunks_embeddings, query_vector)

    ranked_scores = []

    for index, score in enumerate(scores):

        ranked_scores.append((score, index))

    ranked_scores = sorted(
        ranked_scores,
        key=lambda x: x[0],
        reverse=True
    )

    return ranked_scores[:TOP_K]

# ----------------------------
# LLM Response
# ----------------------------

def generate_answer(prompt_input):

    if llm_model == "ai":

        response = ai.generate_text(prompt_input)

        answer_text = (
            response.text
            if hasattr(response, "text")
            else str(response)
        )

    return answer_text

# ----------------------------
# Build System
# ----------------------------

text = extract_text(pdf_file)

chunks, chunks_embeddings = make_chunks(text)

print("✅ Ready. Ask questions. Type 'exit' to stop.\n")

# ----------------------------
# Chat Loop
# ----------------------------

while True:

    user_query_chat = input("Ask a query to your assistant : ")

    if user_query_chat.lower() in ["exit", "quit", "stop", "break"]:

        print("exiting")

        break

    memory_data_dict = load_memory(memory_file)

    # ----------------------------
    # Query Expansion
    # ----------------------------

    expanded_query = expand_query(user_query_chat)

    # ----------------------------
    # Query Embedding
    # ----------------------------

    query_embedding = transformer_model.encode(
        [expanded_query],
        normalize_embeddings=True
    )

    query_vector = query_embedding[0]

    # ----------------------------
    # Retrieval
    # ----------------------------

    top_scores = cosine_search(
        query_vector,
        chunks_embeddings
    )

    print("\nTop retrieved chunks:\n")

    for rank, (score, index) in enumerate(top_scores, start=1):

        print(f"Rank: {rank}")

        print(f"Score: {score:.4f}")

        print(f"Chunk ID: {chunks[index]['chunk_id']}")

        print("\nChunk Preview:\n")

        print(chunks[index]["text"][:500])

        print("\n" + "-" * 60)

    # ----------------------------
    # Hallucination Control
    # ----------------------------

    best_score = top_scores[0][0]

    if best_score < MIN_SIMILARITY_THRESHOLD:

        print("\nAnswer:\n Evidence is weak or missing.\n")

        continue

    # ----------------------------
    # Context Builder
    # ----------------------------

    context_to_current_chat_retrieve = ""

    for score, index in top_scores:

        if score >= MIN_SIMILARITY_THRESHOLD:

            context_to_current_chat_retrieve += f"""

Chunk ID: {chunks[index]["chunk_id"]}

Similarity Score: {score:.4f}

{chunks[index]["text"]}

"""

    # ----------------------------
    # Recent Chat History
    # ----------------------------

    recent_3_chat_text = ""

    for query, ans in memory_data_dict["chat_history"][-3:]:

        recent_3_chat_text += f"""
Q: {query}
A: {ans}

"""

    # ----------------------------
    # Facts Memory
    # ----------------------------

    query_lower = user_query_chat.lower()

    if (
        "favorite" in query_lower
        or "i like" in query_lower
    ):

        if user_query_chat not in memory_data_dict["facts"]["preferences"]:

            memory_data_dict["facts"]["preferences"].append(
                user_query_chat
            )

    if (
        "my name is" in query_lower
        or "i am" in query_lower
    ):

        if user_query_chat not in memory_data_dict["facts"]["user_info"]:

            memory_data_dict["facts"]["user_info"].append(
                user_query_chat
            )

    # ----------------------------
    # Prompt
    # ----------------------------

    prompt_current_chat = f"""
You are a precise AI document assistant.

Your task:
- Answer ONLY from retrieved context
- Do NOT invent information
- Do NOT hallucinate
- If evidence is weak or missing, clearly say:
  "Evidence is weak or missing."

Rules:
1. Use ONLY retrieved context
2. Do not use outside knowledge
3. Be concise
4. If unsure, say evidence missing
5. Mention uncertainty honestly

Facts:
Preferences:
{memory_data_dict["facts"]["preferences"]}

User Info:
{memory_data_dict["facts"]["user_info"]}

Recent Conversation:
{recent_3_chat_text}

Retrieved Context:
{context_to_current_chat_retrieve}

Question:
{user_query_chat}

Answer:
"""

    # ----------------------------
    # Generate Answer
    # ----------------------------

    answer_text = generate_answer(prompt_current_chat)

    print("\nAnswer:\n")

    print(answer_text)

    # ----------------------------
    # Save Chat
    # ----------------------------

    memory_data_dict["chat_history"].append(
        (user_query_chat, answer_text)
    )

    with open(memory_file, 'w') as f:

        json.dump(memory_data_dict, f, indent=2)

    # ----------------------------
    # Long-Term Memory Summary
    # ----------------------------

    if len(memory_data_dict["chat_history"]) > 5:

        old_chat_context = ""

        for ques, ans in memory_data_dict["chat_history"][:-3]:

            old_chat_context += f"""
Question: {ques}

Answer: {ans}

"""

        summarize_prompt = f"""
You are a memory compression system.

Summarize ONLY important long-term knowledge.

Keep:
- important discussions
- key answers
- unresolved questions

Remove:
- repeated content
- greetings
- small talk

Previous Memory:
{memory_data_dict["memory_summary"]}

New Conversations:
{old_chat_context}

Updated Memory:
"""

        memory_summary = generate_answer(
            summarize_prompt
        )

        memory_data_dict["memory_summary"] = memory_summary

        memory_data_dict["chat_history"] = (
            memory_data_dict["chat_history"][-3:]
        )

        with open(memory_file, 'w') as f:

            json.dump(memory_data_dict, f, indent=2)
