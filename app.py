import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import pipeline
# import numpy as np
import tempfile
import streamlit as st


# STREAMLIT PAGE CONFIG
st.set_page_config(
    page_title="Queryo Docs",
    page_icon="🔍",
    layout="centered"
)

# UI STYLING
st.markdown("""
<style>

/* Hide default Streamlit menu text */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* Add custom navbar title */
header[data-testid="stHeader"]::before {
    content: "Queryo Docs";
    font-size: 22px;
    font-weight: 700;
    color: white;
    position: absolute;
    left: 20px;
    top: 11px;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>

/* App background */
.stApp {
    background: linear-gradient(180deg, #0f172a 0%, #020617 100%);
}

/* Chat message bubbles */
[data-testid="stFileUploader"] {
    border-radius: 12px;
    border: 1px solid #334155;
    padding: 8px;
}

/* User message */
[data-testid="stChatMessage-user"] {
    background-color: #1e293b;
}

[data-testid="stChatMessage-assistant"] {
    background-color: #0f172a;
}

/* Assistant message */
[data-testid="stChatMessage"][data-testid="stChatMessage-assistant"] {
    background-color: #0f172a;
}

/* Chat input box */
textarea {
    border-radius: 12px !important;
    border: 1px solid #334155 !important;
}
            
.block-container {
    max-width: 850px;
}

/* Upload box */
[data-testid="stFileUploader"] {
    border-radius: 12px;
}
            
/* Chat input container */
[data-testid="stChatInput"] {
    background-color: transparent;
}

/* Chat input wrapper */
[data-testid="stChatInput"] {
    border: none !important;
}

/* Chat input container */
[data-testid="stChatInput"] > div {
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
}

/* Actual input box */
[data-testid="stChatInput"] textarea {
    background-color: #111827 !important;
    border: 1px solid #374151 !important;
    border-radius: 20px !important;
    padding: 14px !important;
    color: white !important;
    font-size: 15px !important;
    outline: none !important;
    box-shadow: none !important;
}

/* Remove default red outline */
[data-testid="stChatInput"] textarea:focus {
    outline: none !important;
    border: 1px solid #6366f1 !important;
    box-shadow: 0 0 15px rgba(99,102,241,0.35) !important;
    transition: all 0.2s ease;
}

/* Send button */
[data-testid="stChatInput"] button {
    background-color: #6366f1 !important;
    border-radius: 12px !important;
    border: none !important;
}

/* Send button hover */
[data-testid="stChatInput"] button:hover {
    background-color: #4f46e5 !important;
}
            
/* Bottom input container */
[data-testid="stChatInput"] {
    position: fixed;
    bottom: 45px;
    left: 50%;
    transform: translateX(-50%);
    width: 75%;
    max-width: 1000px;
    background: rgba(15, 23, 42, 0.9);
    padding: 17px 20px;
    border-radius: 18px;
    backdrop-filter: blur(8px);
    box-shadow: 0 8px 30px rgba(0,0,0,0.5);
}

/* Chat input textarea */
[data-testid="stChatInput"] textarea {
    border-radius: 25px !important;
    background-color: #0f172a !important;
    border: 1px solid #374151 !important;
}

/* Send button */
[data-testid="stChatInput"] button {
    background-color: #6366f1 !important;
    border-radius: 10px !important;
}
            
.block-container {
    padding-bottom: 140px;
}

[data-testid="stSpinner"] {
    z-index: 9999;
}   

/* Remove focus outline and blue highlight */
[data-testid="stChatInput"] textarea:focus,
[data-testid="stChatInput"] textarea:focus-visible {
    outline: none !important;
    box-shadow: none !important;
}

/* Remove focus ring from the container as well */
[data-testid="stChatInput"]:focus-within {
    outline: none !important;
    box-shadow: none !important;
}        


</style>
""", unsafe_allow_html=True)

# SESSION STATE FOR CHAT HISTORY
# This stores the conversation history
# so messages remain visible after reruns


if "messages" not in st.session_state:
    st.session_state.messages = []


# LOAD LLM MODEL (CACHED)

@st.cache_resource(show_spinner=False)
def load_model():
    return pipeline(
        "text-generation",
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    )

# Loading spinner while model initializes

with st.spinner("Starting Queryo Docs..."):
    generator = load_model()

# APP TITLE

st.markdown("""
<h1 style='text-align:center;'>🔍Queryo Docs</h1>
<p style='text-align:center;color:gray;font-size:18px;'>
Ask anything from your PDFs
</p>
""", unsafe_allow_html=True)

# PDF UPLOAD SECTION

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

# PROCESS PDF AFTER UPLOAD

if uploaded_file:

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    # Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Split text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    docs = text_splitter.split_documents(documents)
    texts = [doc.page_content for doc in docs]

    st.success("✅ PDF uploaded successfully!")

    # Embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts)


    # DISPLAY CHAT HISTORY

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Question input (ONLY AFTER PDF UPLOAD)
  
    question = st.chat_input("Ask a question about the PDF")

    if question:

        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

         # CONVERT QUESTION INTO EMBEDDING

        query_embedding = model.encode([question])

         # SIMILARITY SEARCH

        similarities = cosine_similarity(query_embedding, embeddings)
        top_indices = similarities[0].argsort()[-5:][::-1]

        context = " ".join([texts[i] for i in top_indices])

        
         # BUILD CONVERSATION HISTORY

        history = ""

        recent_messages = st.session_state.messages[-1:]

        for msg in recent_messages:
            role = msg["role"]
            content = msg["content"]

            if role == "user":
                history += f"User: {content}\n"
            else:
                history += f"Assistant: {content}\n"


         # CONTEXT-AWARE SEARCH QUERY
        search_query = history + "\nUser: " + question

        query_embedding = model.encode([search_query])

        similarities = cosine_similarity(query_embedding, embeddings)
        top_indices = similarities[0].argsort()[-5:][::-1]

        context = "\n\n".join([texts[i] for i in top_indices])


        # LLM
        generator = load_model()


        prompt = f"""
You are an assistant answering questions from a document.

Answer ONLY using the information in the context.

If the exact answer is not present in the context,
respond with: "The document does not contain this information."

Context:
{context}

Question:
{question}

Answer:
"""
         # GENERATE ANSWER USING LLM
        with st.spinner("🔍 Analyzing PDF and generating response..."):
            response = generator(
                prompt,
                max_new_tokens=250,
                do_sample=True,
                temperature=0.2,
                top_p=0.9,
                repetition_penalty=1.1
            )

        generated_text = response[0]["generated_text"]
        answer = generated_text.replace(prompt, "").strip()

        # Remove prompt artifacts if model repeats them
        answer = answer.split("Question:")[0].strip()
        answer = answer.split("Given material")[0].strip()

        # Fix capitalization
        if answer:
            answer = answer[0].upper() + answer[1:]

        # Show AI message
        with st.chat_message("assistant"):
            st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})

    