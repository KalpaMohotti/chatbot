from flask import Flask, request, jsonify
import os
from werkzeug.utils import secure_filename
import numpy as np
import faiss
import google.generativeai as genai
from PyPDF2 import PdfReader
import logging
from dotenv import load_dotenv
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_talisman import Talisman

# Load environment variables from .env file
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"pdf"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure the API key for Google Generative AI
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize Flask-Talisman for security headers
Talisman(app)

# Initialize Flask-Limiter for rate limiting
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"]
)

# Global variables
vector_store = None  # Stores the FAISS index
text_chunks = []     # Stores the text chunks extracted from the PDF

# Helper function to check file extension
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to extract text from a PDF
def get_pdf_text(pdf_file):
    text = ""
    try:
        pdf_reader = PdfReader(pdf_file)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
    return text

# Function to split text into chunks
def get_chunks(text, chunk_size=1000, chunk_overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap
    return chunks

# Function to get embeddings using Google Generative AI
def get_gemini_embeddings(texts):
    embeddings = []
    for text in texts:
        try:
            response = genai.embed_content(
                model="models/text-embedding-004",
                content=text,
                task_type="retrieval_document"
            )
            embeddings.append(response['embedding'])
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
    return np.array(embeddings)

# Function to create a FAISS vector store
def create_vector_store(text_chunks):
    try:
        embeddings = get_gemini_embeddings(text_chunks)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        return index
    except Exception as e:
        logger.error(f"Error creating FAISS vector store: {e}")
        return None

# Endpoint to upload and process a PDF file
@app.route("/upload", methods=["POST"])
def upload_file():
    global vector_store, text_chunks

    try:
        if "file" not in request.files:
            logger.error("No file part in the request")
            return jsonify({"error": "No file part"}), 400

        file = request.files["file"]
        if file.filename == "":
            logger.error("No selected file")
            return jsonify({"error": "No selected file"}), 400

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)

            # Extract text from the PDF
            raw_text = get_pdf_text(file_path)

            # Split text into chunks
            text_chunks = get_chunks(raw_text)

            # Create a FAISS vector store
            vector_store = create_vector_store(text_chunks)

            if vector_store is None:
                return jsonify({"error": "Failed to create vector store"}), 500

            logger.info("PDF file processed and embeddings generated successfully")
            return jsonify({"message": "PDF file processed and embeddings generated."}), 200

        logger.error("Invalid file type")
        return jsonify({"error": "Invalid file type"}), 400

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return jsonify({"error": str(e)}), 500

# Endpoint to ask a question
@app.route("/ask", methods=["POST"])
@limiter.limit("10 per minute")
def ask_question():
    global vector_store, text_chunks

    try:
        data = request.json
        user_question = data.get("question")

        if not user_question:
            logger.error("No question provided")
            return jsonify({"error": "Please enter a question."}), 400

        if not vector_store:
            logger.error("PDF file has not been processed")
            return jsonify({"error": "The PDF file has not been processed."}), 400

        if not text_chunks:
            logger.error("No text chunks found")
            return jsonify({"error": "No text chunks found. Please upload and process a PDF file first."}), 400

        # Get the embedding for the user's question
        query_embedding = get_gemini_embeddings([user_question])

        # Search for the most relevant document
        distances, indices = vector_store.search(query_embedding, k=1)

        # Check if any results were found
        if indices.size == 0 or distances[0][0] > 1.0:
            return jsonify({"answer": "I'm sorry, I don't have enough information to answer that. Please ask a different question."}), 200

        # Retrieve the most relevant document
        retrieved_doc = text_chunks[indices[0][0]]

        # Define a system instruction for the chatbot persona
        # IMPORTANT: Change 'Your Name' to your actual name.
        system_instruction = (
            "You are a professional chatbot representing Kalpa Mohotti. "
            "Answer questions about your skills, projects, and professional experience "
            "based on the provided document. Use a first-person perspective ('I', 'me')."
        )

        # Create a RAG prompt with the system instruction
        model = genai.GenerativeModel(
            "gemini-2.5-flash-preview-05-20",
            system_instruction=system_instruction
        )

        rag_prompt = f"Based on the following retrieved information, answer the query:\n\nRetrieved Info: {retrieved_doc}\n\nQuery: {user_question}"
        
        response = model.generate_content(rag_prompt)

        return jsonify({"answer": response.text}), 200

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
