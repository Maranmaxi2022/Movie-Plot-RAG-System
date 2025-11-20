"""
Text chunking module for splitting movie plots into manageable pieces.
"""
from typing import List, Dict
import pandas as pd


class TextChunker:
    """Handles chunking of movie plot texts."""

    def __init__(self, chunk_size: int = 300):
        """
        Initialize the text chunker.

        Args:
            chunk_size: Maximum number of words per chunk
        """
        self.chunk_size = chunk_size

    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks based on word count.

        Args:
            text: The text to chunk

        Returns:
            List of text chunks
        """
        words = text.split()

        if len(words) <= self.chunk_size:
            return [text]

        chunks = []
        for i in range(0, len(words), self.chunk_size):
            chunk = ' '.join(words[i:i + self.chunk_size])
            chunks.append(chunk)

        return chunks

    def chunk_dataframe(self, df: pd.DataFrame) -> List[Dict[str, str]]:
        """
        Chunk all movie plots in a DataFrame.

        Args:
            df: DataFrame with 'Title' and 'Plot' columns

        Returns:
            List of dictionaries with chunked data, including metadata
        """
        chunked_data = []

        for idx, row in df.iterrows():
            title = row['Title']
            plot = row['Plot']

            # Get chunks for this plot
            chunks = self.chunk_text(plot)

            # Create a record for each chunk with metadata
            for chunk_idx, chunk in enumerate(chunks):
                chunked_data.append({
                    'movie_id': idx,
                    'title': title,
                    'chunk_id': chunk_idx,
                    'total_chunks': len(chunks),
                    'text': chunk,
                    'full_plot': plot  # Keep reference to full plot
                })

        print(f"Created {len(chunked_data)} chunks from {len(df)} movies")

        return chunked_data

    def get_context_window(
        self,
        chunks: List[Dict[str, str]],
        chunk_indices: List[int],
        window_size: int = 1
    ) -> List[Dict[str, str]]:
        """
        Get chunks with surrounding context.

        Args:
            chunks: List of all chunks
            chunk_indices: Indices of chunks to retrieve
            window_size: Number of chunks to include before/after

        Returns:
            List of chunks with context
        """
        context_chunks = []
        added_indices = set()

        for idx in chunk_indices:
            # Add surrounding chunks from the same movie
            movie_id = chunks[idx]['movie_id']

            # Find all chunks from the same movie
            movie_chunks = [
                (i, c) for i, c in enumerate(chunks)
                if c['movie_id'] == movie_id
            ]

            # Find the position of current chunk within movie chunks
            chunk_positions = {i: pos for pos, (i, _) in enumerate(movie_chunks)}
            current_pos = chunk_positions.get(idx, 0)

            # Get chunks within window
            start_pos = max(0, current_pos - window_size)
            end_pos = min(len(movie_chunks), current_pos + window_size + 1)

            for pos in range(start_pos, end_pos):
                chunk_idx = movie_chunks[pos][0]
                if chunk_idx not in added_indices:
                    context_chunks.append(chunks[chunk_idx])
                    added_indices.add(chunk_idx)

        return context_chunks
