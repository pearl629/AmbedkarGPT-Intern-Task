 
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_community.document_loaders import TextLoader
from langchain_core.tools import tool
import os
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

# Global vector DB instance
db = None

def initialize_rag_system():
    
    print("Initializing system...")
    
    global db

    # Get the directory where this script is located
    base_dir = Path(__file__).parent

    # Build path to speech.txt
    speech_path = base_dir / "speech.txt"

    # Load text
    loader = TextLoader(str(speech_path))
    documents = loader.load()
    text_content = documents[0].page_content
    # Split into chunks
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=160,
        chunk_overlap=100
    )
    chunks = splitter.split_text(text_content)
    
    # Embeddings
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Chroma vector DB (auto-persisted)
    db = Chroma(
        collection_name="my_collection",
        embedding_function=emb,
        persist_directory="chroma_db"
    )
    
    # Add chunks
    db.add_texts(chunks)
    
    print("Ready!\n")

@tool
def answer_draft(query: str) -> str:
    """Generate a well-structured answer based on the provided research results and query."""
    research_results = db.similarity_search(query, k=3)
    print(research_results)
    
    model = ChatOllama(
        model="phi",
        temperature=0
    )
    
    # Format the context cleanly
    context = "\n\n".join([doc.page_content for doc in research_results])
    
    response = model.invoke([
        {"role": "system", "content": "Generate a clear, concise, and meaningful answer based solely on the provided context."},
        {"role": "user", "content": f"User query: {query}\n\nContext:\n{context}"}
    ])
     
    return response.content

def run_qa_loop():
    """Interactive Q&A loop."""
    model = ChatOllama(
        model="mistral",
        temperature=0
    )
    
    tools = [answer_draft]
    checkpointer = MemorySaver()
    app = create_react_agent(model, tools, checkpointer=checkpointer)
    
    print("Ask questions about the speech. Type 'quit' to exit.\n")
    
    while True:
        query = input("Query: ").strip()
        
        if query.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        
        if not query:
            continue
        
        try:
              
            final_state = app.invoke(
                {"messages": [{"role": "user", "content": query}]},
                config={"configurable": {"thread_id": 42}}
            )
            answer = final_state["messages"][-1].content
            print(f"\nAnswer: {answer}\n")
             
        except Exception as e:
            print(f"Error: {e}\n")
            import traceback
            traceback.print_exc()
        

def main():
    try:
        initialize_rag_system()
        run_qa_loop()
    except FileNotFoundError:
        print("Error: speech.txt file not found!")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()