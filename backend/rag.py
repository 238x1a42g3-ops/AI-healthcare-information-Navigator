import os
import re
import sys
import pdfplumber
import chromadb
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

from dotenv import load_dotenv
from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

PROJECT_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

ENV_PATH = os.path.join(PROJECT_DIR, ".env")

if not os.path.isfile(ENV_PATH):
    ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")

load_dotenv(ENV_PATH)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ============================================================
# GROQ CLIENT
# ============================================================

groq_client = (
    Groq(api_key=GROQ_API_KEY)
    if GROQ_API_KEY
    else None
)


# ============================================================
# CHROMADB SETUP
# ============================================================

print("\n🔄 Starting ChromaDB...")

client = chromadb.PersistentClient(
    path=os.path.join(PROJECT_DIR, "vectorstore")
)

COLLECTION_NAME = "healthcare_report_tfidf"

# Important:
# embedding_function=None prevents ChromaDB from trying
# to automatically use all-MiniLM-L6-v2.

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=None
)

print("✅ ChromaDB started successfully!")


# ============================================================
# GLOBAL VARIABLES
# ============================================================

vectorizer = None
chunk_vectors = None
stored_chunks = []


# ============================================================
# PDF TEXT EXTRACTION
# ============================================================

def extract_pdf_text(pdf_path):

    print("\n==============================================")
    print("📄 PDF TEXT EXTRACTION")
    print("==============================================")

    print("\n🔄 Reading PDF...")

    if not os.path.exists(pdf_path):
        print("\n❌ PDF file not found!")
        print(f"Path entered: {pdf_path}")
        return None

    try:

        full_text = ""

        with pdfplumber.open(pdf_path) as pdf:

            print(f"📑 Total pages: {len(pdf.pages)}")

            for page_number, page in enumerate(pdf.pages, start=1):

                text = page.extract_text()

                if text:
                    full_text += text + "\n"

        print(
            f"\n✅ Extracted {len(full_text)} characters."
        )

        return full_text

    except Exception as e:

        print("\n❌ Error while reading PDF:")
        print(e)

        return None


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    print("\n🔄 Cleaning PDF text...")

    # Replace multiple spaces
    text = re.sub(r"[ \t]+", " ", text)

    # Replace excessive new lines
    text = re.sub(r"\n+", "\n", text)

    # Remove spaces at beginning/end
    text = text.strip()

    print("✅ Text cleaning completed.")

    return text


# ============================================================
# TEXT CHUNKING
# ============================================================

def create_chunks(
    text,
    chunk_size=500,
    overlap=50
):

    print("\n==============================================")
    print("✂️ TEXT CHUNKING")
    print("==============================================")

    print("\n🔄 Creating chunks...")

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunk_words = words[start:end]

        chunk = " ".join(chunk_words)

        if chunk.strip():
            chunks.append(chunk)

        start += chunk_size - overlap

    print(
        f"✅ Created {len(chunks)} chunks."
    )

    return chunks


# ============================================================
# CREATE TF-IDF VECTORS
# ============================================================

def create_vectors(chunks):

    global vectorizer
    global chunk_vectors
    global stored_chunks

    print("\n==============================================")
    print("🧠 CREATING VECTOR REPRESENTATIONS")
    print("==============================================")

    print("\n🔄 Creating TF-IDF vectors...")

    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        stop_words="english"
    )

    # Convert chunks into vectors
    chunk_vectors = vectorizer.fit_transform(
        chunks
    )

    # Save chunks in memory
    stored_chunks = chunks

    print("✅ Vector representations created!")

    print(
        f"📐 Vector dimensions: "
        f"{chunk_vectors.shape[1]}"
    )

    return chunk_vectors


# ============================================================
# STORE VECTORS IN CHROMADB
# ============================================================

def store_chunks(chunks):

    global collection
    global chunk_vectors

    print("\n==============================================")
    print("🗄️ STORING DATA IN CHROMADB")
    print("==============================================")

    print("\n🔄 Preparing chunks...")

    print(
        f"📦 Number of chunks: {len(chunks)}"
    )

    # Create unique IDs
    ids = [
        f"healthcare_chunk_{i}"
        for i in range(len(chunks))
    ]

    # Convert sparse TF-IDF matrix
    # into normal Python lists
    embeddings = (
        chunk_vectors
        .toarray()
        .astype(float)
        .tolist()
    )

    print(
        "\n🔄 Sending TF-IDF vectors to ChromaDB..."
    )

    try:

        # TF-IDF dimensions depend on the current PDF's vocabulary.
        client.delete_collection(
            name=COLLECTION_NAME
        )
        collection = client.create_collection(
            name=COLLECTION_NAME,
            embedding_function=None
        )

        collection.upsert(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=[
                {
                    "source": "healthcare_pdf",
                    "chunk": i
                }
                for i in range(len(chunks))
            ]
        )

        print(
            "✅ Chunks and TF-IDF vectors "
            "stored in ChromaDB!"
        )

        print(
            f"📦 Stored {len(chunks)} vectors."
        )

    except Exception as e:

        print(
            "\n❌ Error while storing data "
            "in ChromaDB:"
        )

        print(e)

        return False

    return True


# ============================================================
# SEARCH CHROMADB
# ============================================================

def search_chromadb(question):

    global vectorizer

    print("\n==============================================")
    print("🔍 SEARCHING CHROMADB")
    print("==============================================")

    print(
        f"\n❓ Question: {question}"
    )

    if vectorizer is None:

        print(
            "\n❌ Vectorizer is not initialized."
        )

        return []

    # Convert question into TF-IDF vector
    question_vector = vectorizer.transform(
        [question]
    )

    # Convert vector to normal list
    query_embedding = (
        question_vector
        .toarray()
        .astype(float)
        .tolist()
    )

    print(
        "\n🔄 Searching ChromaDB..."
    )

    try:

        results = collection.query(
            query_embeddings=query_embedding,
            n_results=4,
            include=[
                "documents",
                "distances",
                "metadatas"
            ]
        )

        documents = results.get(
            "documents",
            [[]]
        )[0]

        distances = results.get(
            "distances",
            [[]]
        )[0]

        print(
            f"✅ Retrieved "
            f"{len(documents)} relevant chunks!"
        )

        # Display similarity information
        print("\n📊 Retrieved information:")

        for i, (document, distance) in enumerate(
            zip(documents, distances),
            start=1
        ):

            print(
                f"\n--- Chunk {i} ---"
            )

            print(
                f"Distance: {distance:.4f}"
            )

            print(
                document[:300] + "..."
            )

        return documents

    except Exception as e:

        print(
            "\n❌ Error while searching "
            "ChromaDB:"
        )

        print(e)

        return []


# ============================================================
# CREATE CONTEXT
# ============================================================

def create_context(documents):

    if not documents:

        return ""

    context = "\n\n".join(
        [
            f"DOCUMENT CHUNK {i + 1}:\n{doc}"
            for i, doc in enumerate(documents)
        ]
    )

    return context


# ============================================================
# SEND INFORMATION TO GROQ
# ============================================================

def generate_answer(question, context):

    print("\n==============================================")
    print("🤖 GENERATING ANSWER WITH GROQ")
    print("==============================================")

    if not context:

        return (
            "I could not find relevant information "
            "in the provided healthcare document."
        )

    system_prompt = """
You are the AI Healthcare Information Navigator.

Your task is to answer questions using ONLY the
information provided in the document context.

Important rules:

1. Do not use outside knowledge.
2. Do not invent information.
3. Do not diagnose diseases.
4. Do not prescribe medicines.
5. Do not recommend changing medication.
6. Explain healthcare information in simple,
   patient-friendly language.
7. If the answer is not available in the context,
   clearly say:

   "I could not find this information in the
   provided healthcare document."

8. This system is for educational information only.
9. Actual medical decisions should be made with
   a qualified healthcare professional.

Answer the user's question clearly and briefly.
"""

    user_prompt = f"""
DOCUMENT CONTEXT:

{context}


USER QUESTION:

{question}


Please answer the question using only the
document context above.
"""

    try:

        response = groq_client.chat.completions.create(

            model="openai/gpt-oss-120b",

            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],

            temperature=0,

            max_tokens=500
        )

        answer = response.choices[0].message.content

        return answer

    except Exception as e:

        print(
            "\n❌ Error while communicating "
            "with Groq:"
        )

        print(e)

        return (
            "Unable to generate an answer "
            "because of a Groq API error."
        )


# ============================================================
# PROCESS PDF
# ============================================================

def process_pdf(pdf_path):

    # --------------------------------------------
    # STEP 1: Extract PDF text
    # --------------------------------------------

    text = extract_pdf_text(
        pdf_path
    )

    if not text:

        return False

    # --------------------------------------------
    # STEP 2: Clean text
    # --------------------------------------------

    text = clean_text(
        text
    )

    # --------------------------------------------
    # STEP 3: Create chunks
    # --------------------------------------------

    chunks = create_chunks(
        text,
        chunk_size=500,
        overlap=50
    )

    if not chunks:

        print(
            "\n❌ No chunks were created."
        )

        return False

    # --------------------------------------------
    # STEP 4: Create TF-IDF vectors
    # --------------------------------------------

    create_vectors(
        chunks
    )

    # --------------------------------------------
    # STEP 5: Store in ChromaDB
    # --------------------------------------------

    success = store_chunks(
        chunks
    )

    if not success:

        return False

    return True


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    if not GROQ_API_KEY:
        print("\n❌ ERROR: GROQ_API_KEY not found!")
        print("Please create a .env file and add:")
        print("GROQ_API_KEY=your_api_key_here")
        return

    print("\n")
    print("==============================================")
    print("🩺 AI HEALTHCARE INFORMATION NAVIGATOR")
    print("==============================================")

    print(
        """
This system will:

1. Read your healthcare PDF
2. Split it into chunks
3. Create vector representations
4. Store the vectors in ChromaDB
5. Search relevant information
6. Send retrieved information to Groq
7. Display the final answer
"""
    )

    # --------------------------------------------
    # Ask PDF path
    # --------------------------------------------

    pdf_path = input(
        "📁 Enter PDF path: "
    ).strip()

    # Remove quotation marks if user enters
    # a path inside quotes
    pdf_path = pdf_path.strip('"').strip("'")

    # --------------------------------------------
    # Process PDF
    # --------------------------------------------

    success = process_pdf(
        pdf_path
    )

    if not success:

        print(
            "\n❌ PDF processing failed."
        )

        return

    # --------------------------------------------
    # Processing completed
    # --------------------------------------------

    print("\n")
    print("==============================================")
    print("✅ PDF PROCESSING COMPLETED")
    print("==============================================")

    print(
        """
Your healthcare document has been:

✔ Extracted
✔ Cleaned
✔ Split into chunks
✔ Converted into TF-IDF vectors
✔ Stored in ChromaDB

You can now ask questions.
"""
    )

    print(
        "Type 'exit' to close the program."
    )

    # --------------------------------------------
    # QUESTION LOOP
    # --------------------------------------------

    while True:

        print("\n----------------------------------------------")

        question = input(
            "\n🩺 Ask a healthcare question: "
        ).strip()

        # Empty question
        if not question:

            print(
                "⚠️ Please enter a question."
            )

            continue

        # Exit
        if question.lower() == "exit":

            print(
                "\n👋 Thank you for using "
                "AI Healthcare Information Navigator!"
            )

            break

        # ----------------------------------------
        # Search ChromaDB
        # ----------------------------------------

        documents = search_chromadb(
            question
        )

        # ----------------------------------------
        # Create context
        # ----------------------------------------

        context = create_context(
            documents
        )

        # ----------------------------------------
        # Generate Groq answer
        # ----------------------------------------

        answer = generate_answer(
            question,
            context
        )

        # ----------------------------------------
        # Display final answer
        # ----------------------------------------

        print("\n")
        print("==============================================")
        print("💡 FINAL ANSWER")
        print("==============================================")

        print(answer)

        print(
            "\n⚠️ Educational information only. "
            "Consult a qualified healthcare professional "
            "for medical decisions."
        )


# ============================================================
# PROGRAM START
# ============================================================

if __name__ == "__main__":

    main()