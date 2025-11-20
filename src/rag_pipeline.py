"""
RAG pipeline that combines retrieval and generation.
"""
import json
from typing import List, Dict, Any, Optional
from .vector_store import VectorStore
from .llm_client import LLMClient


class RAGPipeline:
    """Main RAG pipeline for question answering."""

    def __init__(
        self,
        vector_store: VectorStore,
        llm_client: LLMClient,
        top_k: int = 3
    ):
        """
        Initialize the RAG pipeline.

        Args:
            vector_store: Vector store for retrieval
            llm_client: LLM client for generation
            top_k: Number of chunks to retrieve
        """
        self.vector_store = vector_store
        self.llm_client = llm_client
        self.top_k = top_k

    def _format_contexts(self, results: List[tuple]) -> List[str]:
        """
        Format retrieved contexts for display.

        Args:
            results: List of (chunk, score) tuples

        Returns:
            List of formatted context strings
        """
        contexts = []
        seen_movies = set()

        for chunk, score in results:
            title = chunk['title']
            text = chunk['text']

            # Create a brief context string
            if title not in seen_movies:
                context = f"{title}: {text[:200]}..."
            else:
                context = f"{title} (continued): {text[:150]}..."

            contexts.append(context)
            seen_movies.add(title)

        return contexts

    def _build_prompt(self, query: str, contexts: List[str]) -> str:
        """
        Build the prompt for the LLM.

        Args:
            query: User question
            contexts: Retrieved contexts

        Returns:
            Formatted prompt
        """
        context_text = "\n\n".join([
            f"Context {i+1}:\n{ctx}"
            for i, ctx in enumerate(contexts)
        ])

        prompt = f"""Based on the following movie plot contexts, answer the question.

Movie Plot Contexts:
{context_text}

Question: {query}

Please provide:
1. A natural language answer to the question
2. Reference which movie(s) you used to form your answer
3. If the contexts don't contain enough information to answer fully, say so

Answer:"""

        return prompt

    def _build_system_prompt(self) -> str:
        """
        Build the system prompt for the LLM.

        Returns:
            System prompt string
        """
        return """You are a helpful assistant that answers questions about movies based on their plot summaries.
Be concise but informative. Always cite which movie(s) you're referencing.
If the provided contexts don't contain enough information to answer the question, be honest about it."""

    def query(self, question: str) -> Dict[str, Any]:
        """
        Process a question through the RAG pipeline.

        Args:
            question: User question

        Returns:
            Dictionary with answer, contexts, and reasoning
        """
        print(f"\n🔍 Processing query: {question}")

        # Step 1: Retrieve relevant contexts
        print(f"Retrieving top {self.top_k} relevant chunks...")
        results = self.vector_store.search(question, top_k=self.top_k)

        if not results:
            return {
                "answer": "No relevant information found in the database.",
                "contexts": [],
                "reasoning": "The vector store returned no results for this query."
            }

        # Step 2: Format contexts
        contexts = self._format_contexts(results)

        # Get full context details for reasoning
        context_details = []
        for chunk, score in results:
            context_details.append({
                "movie": chunk['title'],
                "score": f"{score:.3f}",
                "text": chunk['text'][:150] + "..."
            })

        # Step 3: Build prompt and generate answer
        print("Generating answer with LLM...")
        prompt = self._build_prompt(question, contexts)
        system_prompt = self._build_system_prompt()

        answer = self.llm_client.generate(prompt, system_prompt)

        # Step 4: Build reasoning
        movie_titles = list(set([chunk['title'] for chunk, _ in results]))
        reasoning = (
            f"Retrieved {len(results)} relevant plot chunks from {len(movie_titles)} movie(s): "
            f"{', '.join(movie_titles)}. Used these contexts to generate the answer with the LLM."
        )

        # Step 5: Return structured output
        result = {
            "answer": answer.strip(),
            "contexts": contexts,
            "reasoning": reasoning,
            "metadata": {
                "retrieved_movies": movie_titles,
                "num_chunks": len(results),
                "retrieval_scores": [f"{score:.3f}" for _, score in results]
            }
        }

        return result

    def query_json(self, question: str) -> str:
        """
        Process a question and return JSON output.

        Args:
            question: User question

        Returns:
            JSON string with answer, contexts, and reasoning
        """
        result = self.query(question)

        # Create simplified output format
        output = {
            "answer": result["answer"],
            "contexts": result["contexts"],
            "reasoning": result["reasoning"]
        }

        return json.dumps(output, indent=2)

    def batch_query(self, questions: List[str]) -> List[Dict[str, Any]]:
        """
        Process multiple questions.

        Args:
            questions: List of questions

        Returns:
            List of result dictionaries
        """
        results = []
        for i, question in enumerate(questions, 1):
            print(f"\n{'='*60}")
            print(f"Question {i}/{len(questions)}")
            print(f"{'='*60}")
            result = self.query(question)
            results.append(result)

        return results
