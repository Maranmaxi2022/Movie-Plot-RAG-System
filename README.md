# Movie Plot RAG System

A lightweight Retrieval-Augmented Generation (RAG) system that answers questions about movie plots using semantic search and large language models.

## Overview

This system demonstrates a complete RAG pipeline that:
- Loads movie plots from the Wikipedia Movie Plots dataset
- Chunks text into ~300 word segments for better retrieval
- Creates semantic embeddings using Sentence Transformers
- Stores vectors in FAISS for fast similarity search
- Retrieves top-k most relevant chunks for queries
- Generates natural language answers using OpenAI GPT models
- Returns structured JSON output with answer, contexts, and reasoning

## Architecture

The system operates in two main phases:

### Initialization Phase

![Initialization Phase](res/01.png)

The initialization phase prepares the RAG pipeline:

1. **data_loader.py** - Loads the Kaggle dataset and creates a DataFrame with movies and plots
2. **chunker.py** - Splits text into semantic chunks (text segments) with metadata
3. **vector_store.py** - Generates embeddings and builds a FAISS index
4. **llm_client.py** - Initializes the API connection for the language model

Once complete, the RAG pipeline is ready to handle queries.

### Query Phase

![Query Phase](res/02.png)

The query phase handles user questions:

1. **rag_pipeline.py** - Embeds the user question
2. **vector_store.py** - Searches FAISS for top-k most relevant chunks
3. **rag_pipeline.py** - Builds a prompt combining the question with retrieved context
4. **llm_client.py** - Calls the GPT API to generate an answer
5. **Parse Response** - Extracts answer, contexts, reasoning, and metadata
6. **Display to User** - Returns the structured response

## Prerequisites

- Python 3.8 or higher
- OpenAI API key
- Internet connection (for downloading the Kaggle dataset on first run)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Movie-Plot-RAG-System
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your OpenAI API key
# OPENAI_API_KEY=your_openai_api_key_here
```

Alternatively, export the API key directly:
```bash
export OPENAI_API_KEY=your_openai_api_key_here
```

## Usage

### Interactive Mode

Run the system in interactive mode to ask multiple questions:

```bash
python main.py --interactive
```

### Single Query Mode

Ask a single question and get a JSON response:

```bash
python main.py --query "Which movies have crime storylines?"
```

### Advanced Options

Customize the system behavior with various options:

```bash
# Use a specific OpenAI model
python main.py --model gpt-4 --query "What movies involve dreams?"

# Adjust chunk size and retrieval parameters
python main.py --chunk-size 400 --top-k 5 --max-rows 300

# Get JSON output
python main.py --query "Your question here" --json-output
```

### Command Line Options

- `--dataset-path PATH` - Path to existing dataset (skips download)
- `--max-rows N` - Maximum number of movies to load (default: 500)
- `--chunk-size N` - Number of words per chunk (default: 300)
- `--top-k N` - Number of chunks to retrieve (default: 3)
- `--model MODEL` - Specific OpenAI model to use (default: gpt-4o-mini)
- `--query "QUESTION"` - Question to ask (if not provided, enters interactive mode)
- `--interactive` - Run in interactive mode
- `--json-output` - Output results in JSON format

## Test Results

The system has been tested with various queries to demonstrate its capabilities:

### Test 1: Crime Storyline Query

**Query:** "Which movies have crime storylines?"

**Result:**
- **Answer:** The movies with crime storylines are "The Musketeers of Pig Alley," which involves a gangster stealing a wallet, and "The Other Side of the Door," which deals with a man falsely accused of murder.
- **Retrieved Movies:** The Musketeers of Pig Alley, The Other Side of the Door, The Suburbanite
- **Contexts Used:** 3 relevant plot chunks

See full result in [test_01_result.json](test_01_result.json)

### Test 2: Out-of-Domain Query

**Query:** "Who is the current president of Sri Lanka?"

**Result:**
- **Answer:** I cannot provide the current president of Sri Lanka based on the given movie plot contexts. None of the movies mentioned (Disraeli, The Arab, The Lotus Eater) contain information relevant to answering this question. Therefore, I am unable to answer your question.
- **Retrieved Movies:** Disraeli, The Arab, The Lotus Eater
- **Contexts Used:** 3 relevant plot chunks

See full result in [test_02_result.json](test_02_result.json)

**Key Observation:** The system correctly identifies when a query is outside the scope of the movie plot database and explicitly states it cannot answer the question based on available context.