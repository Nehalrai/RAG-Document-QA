import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

embedder = SentenceTransformer('all-MiniLM-L6-v2')

qdrant = QdrantClient(
    host=os.getenv('QDRANT_HOST', 'localhost'),
    port=int(os.getenv('QDRANT_PORT', 6333))
)


def setup_collection(collection_name: str):
    """
    Creates a fresh collection (like a table) in Qdrant to store our vectors.
    size=384 matches the output size of all-MiniLM-L6-v2.
    Distance.COSINE means similarity is measured by the angle between vectors.
    """
    if qdrant.collection_exists(collection_name):
        qdrant.delete_collection(collection_name)
    qdrant.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )


def load_document(file_path: str):
    """
    Reads a file from disk and returns it as a list of LangChain Document objects.
    Each Document has a page_content (the text) and metadata (filename, page number etc).
    Supports PDF, DOCX, and plain TXT files.
    """
    if file_path.endswith('.pdf'):
        return PyPDFLoader(file_path).load()
    elif file_path.endswith('.docx'):
        return Docx2txtLoader(file_path).load()
    else:
        with open(file_path, encoding='utf-8') as f:
            return [Document(page_content=f.read())]


def ingest(file_path: str, collection_name: str = 'docs'):
    """
    Full ingestion pipeline:
    1. Load the document from disk
    2. Split it into overlapping chunks
    3. Convert each chunk into a vector (embedding)
    4. Store the vector + original text + metadata in Qdrant
    """
    # Step 1: Load
    docs = load_document(file_path)

    # Step 2: Chunk
    # chunk_size=500: each chunk is at most 500 characters
    # chunk_overlap=50: the last 50 characters of one chunk are repeated
    #   at the start of the next — this prevents losing context at boundaries
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    chunks = splitter.split_documents(docs)

    # Step 3 + 4: Embed each chunk and build Qdrant points
    points = []
    for i, chunk in enumerate(chunks):
        # embedder.encode() runs the neural network locally and returns a numpy array
        # .tolist() converts it to a plain Python list that Qdrant accepts
        vector = embedder.encode(chunk.page_content).tolist()

        points.append(PointStruct(
            id=i,                          # unique ID for this point in Qdrant
            vector=vector,                 # the 384-dimensional embedding
            payload={                      # metadata stored alongside the vector
                'text': chunk.page_content,
                'source': file_path,
                'page': chunk.metadata.get('page', 0)
            }
        ))

    # Upsert = insert or update. Sends all points to Qdrant in one call.
    qdrant.upsert(collection_name=collection_name, points=points)
    print(f'Ingested {len(points)} chunks from {file_path}')


if __name__ == '__main__':
    # Quick test: run this file directly to ingest a document
    # Change the path below to any PDF, DOCX, or TXT file you have
    setup_collection('docs')
    ingest(r'D:\DM_LAB_PROJECT\paper1.pdf')
