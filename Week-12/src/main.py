import os

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# --------------------------------------------------
# 1. Sample Document
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
# 2. Configuration
# --------------------------------------------------

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 300
CHUNK_OVERLAP = 30
TOP_K = 2

QUERY = "What are the effects of climate change?"


# --------------------------------------------------
# 3. Split Text into Chunks
# --------------------------------------------------

def create_chunks(text: str):
    """Split the sample document into overlapping chunks."""

    document = Document(
        page_content=text.strip(),
        metadata={"source": "climate_change_sample"},
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )

    chunks = splitter.split_documents([document])

    return chunks


# --------------------------------------------------
# 4. Create Local Embeddings
# --------------------------------------------------

def create_embeddings():
    """Load a local sentence-transformer embedding model."""

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    return embeddings


# --------------------------------------------------
# 5. Build FAISS Vector Store
# --------------------------------------------------

def create_vector_store(chunks, embeddings):
    """Convert chunks into vectors and store them in FAISS."""

    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings,
    )

    return vectorstore


# --------------------------------------------------
# 6. Retrieve Relevant Chunks
# --------------------------------------------------

def retrieve_chunks(vectorstore, query: str):
    """Retrieve the top-k chunks and relevance scores."""

    results = vectorstore.similarity_search_with_relevance_scores(
        query,
        k=min(TOP_K, vectorstore.index.ntotal),
    )

    return results


# --------------------------------------------------
# 7. Main Experiment
# --------------------------------------------------

def main():
    print("=" * 65)
    print("WEEK 12: RAG DATA INDEXING WITH FAISS")
    print("=" * 65)

    print(f"\nEmbedding model: {EMBEDDING_MODEL}")
    print(f"Chunk size: {CHUNK_SIZE}")
    print(f"Chunk overlap: {CHUNK_OVERLAP}")

    try:
        # Split document.
        chunks = create_chunks(SAMPLE_TEXT)

        print(f"\nNumber of chunks created: {len(chunks)}")

        if len(chunks) < 2:
            raise ValueError(
                "The document must produce at least two chunks."
            )

        # Load embedding model.
        print("\nLoading embedding model...")
        embeddings = create_embeddings()

        # Build vector index.
        print("Creating FAISS vector store...")
        vectorstore = create_vector_store(chunks, embeddings)

        print(
            f"Vectors stored in FAISS: {vectorstore.index.ntotal}"
        )

        # Retrieve relevant chunks.
        print(f"\nQuery: {QUERY}")
        print(f"Retrieving top {TOP_K} chunks...")

        results = retrieve_chunks(vectorstore, QUERY)

        print("\n" + "=" * 65)
        print("RETRIEVED RESULTS")
        print("=" * 65)

        for rank, (document, score) in enumerate(results, start=1):
            print(f"\nResult {rank}")
            print(f"Relevance score: {score:.4f}")
            print(
                f"Chunk start index: "
                f"{document.metadata.get('start_index', 'N/A')}"
            )
            print(f"Content:\n{document.page_content}")
            print("-" * 65)

        print("\nExperiment completed successfully.")

    except Exception as error:
        print("\nERROR: Experiment failed.")
        print(f"Details: {error}")
        print(
            "\nCheck package installation, internet connectivity, "
            "and whether the embedding model downloaded successfully."
        )


if __name__ == "__main__":
    main()