 # AmbedkarGPT - RAG Q&A System

A command-line Question & Answer system built using Retrieval-Augmented Generation (RAG) to answer questions based on Dr. B.R. Ambedkar's speech excerpt from "Annihilation of Caste."

 
## Overview

This project demonstrates a functional RAG pipeline that satisfies all assignment requirements:

✅ **1. Load text file** - Loads speech.txt using LangChain's TextLoader  
✅ **2. Split into chunks** - Uses CharacterTextSplitter with configurable chunk size and overlap  
✅ **3. Create embeddings** - Uses HuggingFaceEmbeddings with sentence-transformers/all-MiniLM-L6-v2  
✅ **4. Store in vector database** - Persists embeddings in ChromaDB locally  
✅ **5. Retrieve relevant chunks** - Performs similarity search to find top-k relevant chunks  
✅ **6. Generate answers** - Uses Ollama with Mistral 7B to generate contextual answers  
✅ **7. Conversational memory** - Maintains conversation history across multiple queries using LangGraph's MemorySaver

### Technical Stack (All Assignment Requirements Met)

- **Python**: 3.12.12 (3.8+ required ✅)
- **Framework**: LangChain (required ✅)
- **Vector Store**: ChromaDB (required ✅)
- **Embeddings**: HuggingFace sentence-transformers/all-MiniLM-L6-v2 (required ✅)
- **LLM**: Ollama with Mistral 7B (required ✅)
- **Agent Framework**: LangGraph with MemorySaver for conversation persistence ✅
- **Cost**: 100% free, no API keys, completely local (required ✅)

 
 

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/pearl629/AmbedkarGPT-Intern-Task
cd AmbedkarGPT-Intern-Task
 
 
``` 

### Step 2: Set Up Virtual Environment

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

 

## Usage

### Running the System

1. **Ensure Ollama is running** 
   
 

2. **Activate your virtual environment** (if not already activated):
   ```bash
   venv\Scripts\activate  # Windows
   source venv/bin/activate  # Mac/Linux
   ```

3. **Run the program**:
   ```bash
   python main.py
   ```

4. **Ask questions** about the speech content. The system remembers previous questions and answers in the same session!

5. **Type `quit`, `exit`, or `q`** to stop the program

### Example Conversation

```
Query: What is the main topic of the speech?
Answer: The main topic discusses the caste system and its abolition...

Query: Can you elaborate on that?
Answer: [System remembers context from previous question and provides detailed response]

Query: quit
Goodbye!
```

 

## Project Structure

```
AmbedkarGPT-Intern-Task/
│
├── main.py                  # Main application code
├── speech.txt               # Dr. Ambedkar's speech text
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── .gitignore              # Git ignore rules
│
├── venv/                   # Virtual environment (not in repo)
└── chroma_db/              # ChromaDB storage (created on first run)
```

## Technical Details

### RAG Pipeline

1. **Text Loading**: Loads `speech.txt` using LangChain's TextLoader
2. **Chunking**: Splits text into 200-character chunks with 50-character overlap
3. **Embeddings**: Converts chunks to vectors using HuggingFace's all-MiniLM-L6-v2
4. **Vector Storage**: Stores embeddings in ChromaDB for fast similarity search
5. **Retrieval**: Finds top 3 most relevant chunks for each query
6. **Generation**: Sends context + query to Mistral 7B for answer generation
7. **Memory**: LangGraph's MemorySaver maintains conversation history across queries

### Key Components

- **LangChain**: Orchestrates the entire RAG workflow
- **ChromaDB**: Local vector database for semantic search
- **HuggingFace Embeddings**: Converts text to numerical vectors
- **Ollama + Mistral**: Generates natural language answers
- **LangGraph**: Manages agent workflow with persistent memory using MemorySaver
- **React Agent**: Tool-calling agent that can use the RAG retrieval tool dynamically
 
 
 

 
 