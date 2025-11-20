#!/usr/bin/env python3
"""
Main CLI application for the Movie Plot RAG System.
"""
import os
import sys
import json
import argparse
from dotenv import load_dotenv

from src import DataLoader, TextChunker, VectorStore, get_llm_client, RAGPipeline


def setup_environment():
    """Load environment variables from .env file."""
    load_dotenv()


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Movie Plot RAG System - Ask questions about movie plots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode with custom dataset
  python main.py --interactive

  # Single query mode
  python main.py --query "What movies feature artificial intelligence?"

  # Use sample data (no download)
  python main.py --sample --query "Tell me about space movies"

  # Use OpenAI instead of Anthropic
  python main.py --provider openai --query "What movies involve dreams?"

  # Custom configuration
  python main.py --chunk-size 400 --top-k 5 --max-rows 300
        """
    )

    # Data options
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Use sample data instead of downloading dataset"
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        help="Path to existing dataset (skips download)"
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=int(os.getenv("MAX_DATASET_ROWS", "500")),
        help="Maximum number of movies to load (default: 500)"
    )

    # RAG configuration
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=int(os.getenv("CHUNK_SIZE", "300")),
        help="Number of words per chunk (default: 300)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=int(os.getenv("TOP_K_RESULTS", "3")),
        help="Number of chunks to retrieve (default: 3)"
    )

    # LLM options
    parser.add_argument(
        "--provider",
        type=str,
        choices=["anthropic", "openai"],
        default="anthropic",
        help="LLM provider to use (default: anthropic)"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Specific model to use (optional)"
    )

    # Query options
    parser.add_argument(
        "--query",
        type=str,
        help="Question to ask (if not provided, enters interactive mode)"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--json-output",
        action="store_true",
        help="Output results in JSON format"
    )

    return parser.parse_args()


def print_banner():
    """Print welcome banner."""
    banner = """
╔════════════════════════════════════════════════════════════╗
║           🎬 Movie Plot RAG System 🎬                      ║
║  Ask questions about movie plots using AI-powered search   ║
╚════════════════════════════════════════════════════════════╝
    """
    print(banner)


def initialize_rag_system(args):
    """
    Initialize the RAG system with all components.

    Args:
        args: Parsed command line arguments

    Returns:
        Initialized RAG pipeline
    """
    print("\n" + "="*60)
    print("🚀 Initializing RAG System")
    print("="*60)

    # Step 1: Load data
    print("\n[1/5] Loading movie plot data...")
    data_loader = DataLoader(max_rows=args.max_rows)

    if args.sample:
        print("Using sample data...")
        df = data_loader.get_sample_data()
    elif args.dataset_path:
        print(f"Loading from: {args.dataset_path}")
        df = data_loader.load_data(args.dataset_path)
    else:
        df = data_loader.load_data()

    print(f"✓ Loaded {len(df)} movies")

    # Step 2: Chunk texts
    print(f"\n[2/5] Chunking texts (chunk size: {args.chunk_size} words)...")
    chunker = TextChunker(chunk_size=args.chunk_size)
    chunks = chunker.chunk_dataframe(df)
    print(f"✓ Created {len(chunks)} chunks")

    # Step 3: Build vector store
    print("\n[3/5] Building vector store with FAISS...")
    vector_store = VectorStore()
    vector_store.build_index(chunks)
    print(f"✓ Vector store ready")

    # Step 4: Initialize LLM
    print(f"\n[4/5] Initializing LLM ({args.provider})...")
    try:
        llm_client = get_llm_client(
            provider=args.provider,
            model=args.model
        )
        print(f"✓ LLM client ready")
    except Exception as e:
        print(f"\n❌ Error initializing LLM: {e}")
        print("\nMake sure you have set up your API keys:")
        print("  - For Anthropic: export ANTHROPIC_API_KEY=your_key")
        print("  - For OpenAI: export OPENAI_API_KEY=your_key")
        print("\nOr create a .env file with your API keys")
        sys.exit(1)

    # Step 5: Create RAG pipeline
    print(f"\n[5/5] Creating RAG pipeline (top-k: {args.top_k})...")
    rag_pipeline = RAGPipeline(
        vector_store=vector_store,
        llm_client=llm_client,
        top_k=args.top_k
    )
    print(f"✓ RAG pipeline ready")

    print("\n" + "="*60)
    print("✅ Initialization complete!")
    print("="*60)

    return rag_pipeline


def print_result(result: dict, json_output: bool = False):
    """
    Print the query result.

    Args:
        result: Result dictionary from RAG pipeline
        json_output: Whether to output as JSON
    """
    if json_output:
        output = {
            "answer": result["answer"],
            "contexts": result["contexts"],
            "reasoning": result["reasoning"]
        }
        print(json.dumps(output, indent=2))
    else:
        print("\n" + "="*60)
        print("📝 ANSWER")
        print("="*60)
        print(result["answer"])

        print("\n" + "="*60)
        print("📚 CONTEXTS")
        print("="*60)
        for i, context in enumerate(result["contexts"], 1):
            print(f"\n{i}. {context}")

        print("\n" + "="*60)
        print("🧠 REASONING")
        print("="*60)
        print(result["reasoning"])

        if "metadata" in result:
            print("\n" + "="*60)
            print("📊 METADATA")
            print("="*60)
            print(f"Movies: {', '.join(result['metadata']['retrieved_movies'])}")
            print(f"Chunks: {result['metadata']['num_chunks']}")
            print(f"Scores: {', '.join(result['metadata']['retrieval_scores'])}")

        print("\n" + "="*60)


def interactive_mode(rag_pipeline: RAGPipeline, json_output: bool = False):
    """
    Run the system in interactive mode.

    Args:
        rag_pipeline: Initialized RAG pipeline
        json_output: Whether to output as JSON
    """
    print("\n" + "="*60)
    print("💬 Interactive Mode")
    print("="*60)
    print("Type your questions below. Type 'quit' or 'exit' to stop.")
    print("="*60 + "\n")

    while True:
        try:
            question = input("\n🎬 Your question: ").strip()

            if not question:
                continue

            if question.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break

            result = rag_pipeline.query(question)
            print_result(result, json_output)

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            continue


def main():
    """Main entry point."""
    # Load environment variables
    setup_environment()

    # Parse arguments
    args = parse_arguments()

    # Print banner
    if not args.json_output:
        print_banner()

    # Initialize RAG system
    try:
        rag_pipeline = initialize_rag_system(args)
    except Exception as e:
        print(f"\n❌ Error during initialization: {e}")
        sys.exit(1)

    # Run query or interactive mode
    if args.query:
        # Single query mode
        result = rag_pipeline.query(args.query)
        print_result(result, args.json_output)
    else:
        # Interactive mode
        interactive_mode(rag_pipeline, args.json_output)


if __name__ == "__main__":
    main()
