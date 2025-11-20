"""
Vector store module using FAISS for similarity search.
"""
from typing import List, Dict, Tuple
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


class VectorStore:
    """FAISS-based vector store for semantic search."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the vector store.

        Args:
            model_name: Name of the sentence transformer model to use for embeddings
        """
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks = []
        self.dimension = None

    def build_index(self, chunks: List[Dict[str, str]]) -> None:
        """
        Build FAISS index from text chunks.

        Args:
            chunks: List of chunk dictionaries with 'text' field
        """
        print(f"Building vector index for {len(chunks)} chunks...")

        # Store chunks for later retrieval
        self.chunks = chunks

        # Extract texts
        texts = [chunk['text'] for chunk in chunks]

        # Generate embeddings
        print("Generating embeddings...")
        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            batch_size=32
        )

        # Convert to float32 for FAISS
        embeddings = np.array(embeddings).astype('float32')

        # Get dimension
        self.dimension = embeddings.shape[1]

        # Create FAISS index (using L2 distance)
        self.index = faiss.IndexFlatL2(self.dimension)

        # Add embeddings to index
        self.index.add(embeddings)

        print(f"Index built successfully with {self.index.ntotal} vectors")

    def search(
        self,
        query: str,
        top_k: int = 3
    ) -> List[Tuple[Dict[str, str], float]]:
        """
        Search for similar chunks given a query.

        Args:
            query: Search query text
            top_k: Number of results to return

        Returns:
            List of tuples (chunk_dict, similarity_score)
        """
        if self.index is None or self.index.ntotal == 0:
            raise ValueError("Index not built. Call build_index() first.")

        # Encode query
        query_embedding = self.model.encode([query])
        query_embedding = np.array(query_embedding).astype('float32')

        # Search
        distances, indices = self.index.search(query_embedding, top_k)

        # Prepare results
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.chunks):
                chunk = self.chunks[idx]
                # Convert L2 distance to similarity score (inverse)
                similarity = 1 / (1 + distance)
                results.append((chunk, float(similarity)))

        return results

    def get_embeddings_stats(self) -> Dict[str, any]:
        """
        Get statistics about the vector store.

        Returns:
            Dictionary with statistics
        """
        if self.index is None:
            return {
                'total_vectors': 0,
                'dimension': None,
                'total_chunks': 0
            }

        return {
            'total_vectors': self.index.ntotal,
            'dimension': self.dimension,
            'total_chunks': len(self.chunks),
            'model_name': self.model.get_sentence_embedding_dimension()
        }

    def save_index(self, filepath: str) -> None:
        """
        Save FAISS index to disk.

        Args:
            filepath: Path to save the index
        """
        if self.index is None:
            raise ValueError("No index to save")

        faiss.write_index(self.index, filepath)
        print(f"Index saved to {filepath}")

    def load_index(self, filepath: str) -> None:
        """
        Load FAISS index from disk.

        Args:
            filepath: Path to the saved index
        """
        self.index = faiss.read_index(filepath)
        self.dimension = self.index.d
        print(f"Index loaded from {filepath}")
