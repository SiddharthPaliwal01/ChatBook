import os
import uuid
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import pipeline

# Set up Google API key
os.environ["GOOGLE_API_KEY"] = "your-api-key"

# Initialize DistilBERT for extractive QA
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

# Load and process book
loader = PyPDFLoader("book.pdf")
documents = loader.load()

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

# Add unique IDs to chunks
for chunk in chunks:
    chunk.metadata["id"] = str(uuid.uuid4())

# Create embeddings and store in Chroma
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = Chroma.from_documents(chunks, embeddings)

# Set up Gemini for generative QA
llm = ChatGoogleGenerativeAI(model="gemini-pro")
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Function to determine question type (basic heuristic)
def is_complex_question(question):
    complex_keywords = ["why", "theme", "motivation", "explain"]
    return any(keyword in question.lower() for keyword in complex_keywords)

# Main question-answering function
def answer_question(question):
    # Retrieve relevant chunks
    retrieved_docs = retriever.get_relevant_documents(question)
    context = " ".join([doc.page_content for doc in retrieved_docs])

    # Route based on question type
    if is_complex_question(question):
        # Use Gemini for complex questions
        answer = qa_chain.run(question)
        return {"answer": answer, "source": context}
    else:
        # Use DistilBERT for simple questions
        result = qa_pipeline(question=question, context=context)
        return {"answer": result["answer"], "source": context}

# Test the system
question = "What is the main theme of the book?"
response = answer_question(question)
print(f"Answer: {response['answer']}")
print(f"Source: {response['source']}")