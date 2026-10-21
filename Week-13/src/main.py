import os
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq


# --------------------------------------------------
# 1. Environment Setup
# --------------------------------------------------

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY not found. Check your .env file."
    )

MODEL = "qwen/qwen3.8-27b"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 300
CHUNK_OVERLAP = 30
TOP_K = 2


# --------------------------------------------------
# 2. Sample Document
# --------------------------------------------------

SAMPLE_TEXT = """
Climate change refers to long-term changes in global temperatures
and weather patterns. Human activities, especially burning fossil
fuels such as coal, oil, and gas, are the primary drivers of modern
climate change. These activities release greenhouse gases such as
carbon dioxide into the atmosphere.

One major effect of climate change is rising global temperatures.
Higher temperatures contribute to heatwaves, droughts, and changes
in rainfall patterns. Some regions experience severe water shortages,
while others experience more intense rainfall and flooding.

Climate change also causes sea levels to rise. As glaciers and ice
sheets melt, additional water enters the oceans. Warmer ocean water
also expands, contributing to rising sea levels. Coastal communities
face increased risks from erosion, flooding, and storm surges.

Extreme weather events can threaten agriculture and food security.
Droughts and unpredictable rainfall can reduce crop yields.
Flooding and heat stress can damage agricultural land and affect
livestock. These impacts may increase food prices and threaten
livelihoods.

Climate change affects biodiversity and ecosystems. Many species
must adapt to changing temperatures or move to new habitats.
Coral reefs are particularly vulnerable to warmer oceans and ocean
acidification. Ecosystem disruption can affect food chains and
reduce biodiversity.

Reducing greenhouse gas emissions is essential for limiting future
warming. Renewable energy sources, including solar and wind power,
can reduce dependence on fossil fuels. Improving energy efficiency,
protecting forests, and using sustainable transportation can also
help reduce emissions.

Adaptation is equally important. Coastal protection, drought-resistant
crops, improved water management, and early-warning systems can help
communities prepare for climate impacts. Governments, businesses,
and individuals all have roles in addressing climate change.
"""


# --------------------------------------------------
# 3. Create Document Chunks
# --------------------------------------------------

def create_chunks():
    document = Document(
        page_content=SAMPLE_TEXT.strip(),
        metadata={"source": "climate_change_sample.txt"},
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )

    chunks = splitter.split_documents([document])

    print(f"Number of chunks created: {len(chunks)}")

    return chunks


# --------------------------------------------------
# 4. Initialize Embeddings
# --------------------------------------------------

def create_embeddings():
    print("\nLoading Hugging Face embedding model...")

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


# --------------------------------------------------
# 5. Build FAISS Vector Store
# --------------------------------------------------

def create_vectorstore(chunks, embeddings):
    print("Creating FAISS vector store...")

    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings,
    )

    print(f"Vectors indexed: {vectorstore.index.ntotal}")

    return vectorstore


# --------------------------------------------------
# 6. Initialize Groq LLM
# --------------------------------------------------

def create_llm():
    return ChatGroq(
        model=MODEL,
        temperature=0,
        api_key=GROQ_API_KEY,
        reasoning_effort="none",
    )


# --------------------------------------------------
# 7. Define RAG Prompt
# --------------------------------------------------

RAG_PROMPT = ChatPromptTemplate.from_template(
    """
You are a question-answering assistant.

Answer the question using ONLY the supplied context.

Rules:
1. Do not use outside knowledge.
2. If the context does not contain the answer, say:
   "I don't know based on the provided context."
3. Give a concise, clear answer.
4. Do not invent facts.

Context:
{context}

Question:
{question}

Answer:
"""
)


# --------------------------------------------------
# 8. Format Retrieved Documents
# --------------------------------------------------

def format_docs(documents):
    return "\n\n".join(
        document.page_content
        for document in documents
    )


# --------------------------------------------------
# 9. Build LCEL RAG Chain
# --------------------------------------------------

def create_rag_chain(retriever, llm):
    parser = StrOutputParser()

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | RAG_PROMPT
        | llm
        | parser
    )

    return rag_chain


# --------------------------------------------------
# 10. Ask Questions and Show Sources
# --------------------------------------------------

def answer_question(question, retriever, rag_chain):
    print("\n" + "=" * 70)
    print(f"QUESTION: {question}")
    print("=" * 70)

    # Retrieve source chunks for transparency.
    documents = retriever.invoke(question)

    print("\nRETRIEVED CONTEXT:")

    for index, document in enumerate(documents, start=1):
        print(f"\nChunk {index}:")
        print(document.page_content)

    # Generate grounded answer through LCEL.
    answer = rag_chain.invoke(question)

    print("\nGENERATED ANSWER:")
    print(answer)


# --------------------------------------------------
# 11. Main Experiment
# --------------------------------------------------

def main():
    print("=" * 70)
    print("WEEK 13: RAG CHAIN USING LANGCHAIN LCEL AND GROQ")
    print("=" * 70)

    print(f"\nLLM: {MODEL}")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"Chunk size: {CHUNK_SIZE}")
    print(f"Chunk overlap: {CHUNK_OVERLAP}")
    print(f"Top-k retrieval: {TOP_K}")

    try:
        # Step 1: Split document.
        chunks = create_chunks()

        if not chunks:
            raise ValueError("No document chunks were created.")

        # Step 2: Load embedding model.
        embeddings = create_embeddings()

        # Step 3: Build FAISS index.
        vectorstore = create_vectorstore(chunks, embeddings)

        # Step 4: Create retriever.
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": min(TOP_K, len(chunks))}
        )

        # Step 5: Initialize LLM.
        llm = create_llm()

        # Step 6: Build RAG chain.
        rag_chain = create_rag_chain(retriever, llm)

        # Step 7: Test questions.
        questions = [
            "What causes climate change?",
            "What are the effects of climate change?",
        ]

        for question in questions:
            answer_question(
                question,
                retriever,
                rag_chain,
            )

        # Step 8: Test behavior for an unsupported question.
        answer_question(
            "What is the population of Mars?",
            retriever,
            rag_chain,
        )

        print("\n" + "=" * 70)
        print("WEEK 13 EXPERIMENT COMPLETED")
        print("=" * 70)

    except Exception as error:
        print("\nERROR: Experiment failed.")
        print(f"Details: {error}")

        print(
            "\nCheck your Groq API key, model access, "
            "installed packages, and internet connection."
        )


if __name__ == "__main__":
    main()