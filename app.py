import streamlit as st
import os
import cloudscraper
import html2text
from pinecone import Pinecone, ServerlessSpec # <--- Important import
from llama_index.core import VectorStoreIndex, StorageContext, Document, Settings
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.llms.litellm import LiteLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# API CONFIGURATION
from dotenv import load_dotenv

# Load environment variables from .env file (for local development)
load_dotenv()

# Get the API key
HF_TOKEN = os.getenv("HF_TOKEN")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "pra-rulebook-index"

if HF_TOKEN is None:
    raise ValueError("API key not found. Ensure YOUR_API_KEY_NAME is set in environment variables or .env file.")

os.environ["HUGGINGFACE_API_KEY"] = HF_TOKEN

# MODEL SETTINGS 
@st.cache_resource
def init_settings():
    Settings.llm = LiteLLM(model="huggingface/meta-llama/Llama-3.3-70B-Instruct", api_key=HF_TOKEN)
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.chunk_size = 512

init_settings()

#  PINECONE INITIALIZATION 
pc = Pinecone(api_key=PINECONE_API_KEY)

# Logic to prevent 404 Error
existing_indexes = [i.name for i in pc.list_indexes()]
if PINECONE_INDEX_NAME not in existing_indexes:
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

pinecone_index = pc.Index(PINECONE_INDEX_NAME)
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)



#  SCRAPING ENGINE 
def get_live_content(url):
    """Live scrape a PRA page."""
    try:
        scraper = cloudscraper.create_scraper()
        res = scraper.get(url, timeout=10)
        if res.status_code == 200:
            h = html2text.HTML2Text()
            h.ignore_links = True
            return h.handle(res.text)
    except Exception as e:
        return None
    return None

def sync_rulebook_to_pinecone(url_list):
    """Scrapes live data and upserts to Pinecone."""
    documents = []
    for url in url_list:
        content = get_live_content(url)
        if content:
            
            doc = Document(
                text=content, 
                doc_id=url.replace("/", "_"), 
                metadata={"url": url, "source": "PRA Rulebook"}
            )
            documents.append(doc)
    
    if documents:
        # This sends the data to Pinecone
        VectorStoreIndex.from_documents(
            documents, 
            storage_context=storage_context,
            show_progress=True
        )
        return True
    return False

#  STREAMLIT UI
st.set_page_config(page_title="PRA Pinecone Assistant", layout="wide")
st.title("🌲 PRA Assistant")

# Sidebar for Real-time Updates
with st.sidebar:
    st.header("Live Sync Control")
    st.info("This updates the cloud database with real-time web content.")
    
    # Example: List of key sections to keep updated
    target_urls = [
        "https://www.prarulebook.co.uk/rulebook/Content/Part/211136", # Fundamental Rules
        "https://www.prarulebook.co.uk/rulebook/Content/Part/211153", # Definition of Capital
        "https://www.prarulebook.co.uk/rulebook/Content/Part/211145", # Reporting Pillar 1
    ]
    
    if st.button("🚀 Sync Live Rulebook"):
        with st.spinner("Syncing web content to Pinecone..."):
            success = sync_rulebook_to_pinecone(target_urls)
            if success:
                st.success("Pinecone Index Updated!")
    
    if st.button("🧹 Wipe Index"):
        pinecone_index.delete(delete_all=True)
        st.warning("Index cleared.")

# CHAT INTERFACE
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if query := st.chat_input("Ask a regulatory question..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        # We initialize the index from the existing Pinecone store
        index = VectorStoreIndex.from_vector_store(vector_store)
        query_engine = index.as_query_engine(similarity_top_k=5)
        
        with st.spinner("Querying Pinecone..."):
            response = query_engine.query(query)
            st.markdown(str(response))
            st.session_state.messages.append({"role": "assistant", "content": str(response)})
            
            with st.expander("🔗 Sources Found in Pinecone"):
                for node in response.source_nodes:
                    st.write(f"**URL:** {node.metadata.get('url')} | **Score:** {node.score:.2f}")