from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

CHROMA_PERSIST_DIR = "./chroma_store"

# Initialize Groq for question generation (free)
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)

# Using a standard, fast sentence transformer from Hugging Face
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def setup_rag_retriever(resume_text: str, jd_text: str):
    full_context = f"JOB DESCRIPTION:\n{jd_text}\n\nRESUME:\n{resume_text}"

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_text(full_context)

    vectorstore = Chroma.from_texts(
        texts=splits,
        embedding=embeddings,
        persist_directory=CHROMA_PERSIST_DIR
    )
    return vectorstore.as_retriever()


def load_rag_retriever():
    """Load an already-persisted retriever from disk."""
    vectorstore = Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embeddings
    )
    return vectorstore.as_retriever()


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def generate_interview_questions(retriever, focus_area: str):
    system_prompt = (
        "You are an expert technical interviewer. Use the retrieved context about the "
        "candidate's resume and the job description to generate 3 challenging interview questions "
        "focusing on the provided topic. "
        "If you don't know the answer based on the context, ask a general behavioral question instead."
        "\n\nContext:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Focus area: {focus_area}. Generate the questions now.")
    ])

    rag_chain = (
        {
            "context": RunnableLambda(lambda x: x) | retriever | format_docs,
            "focus_area": RunnableLambda(lambda x: x)
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain.invoke(focus_area)