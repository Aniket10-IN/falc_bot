import streamlit as st
from dotenv import load_dotenv
import os
from query_embeddings import BlogQueryEngine
import time

load_dotenv()

# Configure Streamlit page
st.set_page_config(
    page_title="Blog RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

if "messages" not in st.session_state:
    st.session_state.messages = []

@st.cache_resource
def get_query_engine():
    return BlogQueryEngine(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        db_path="./chroma_db"
    )

def main():
    st.title("📚 Blog Content Q&A Bot")
    
    with st.sidebar:
        st.markdown("### About")
        st.markdown("""
        This chatbot uses RAG (Retrieval Augmented Generation) to answer 
        questions based on blog content. It will:
        1. Search for relevant blog passages
        2. Use them to generate accurate answers
        3. Show you the source of information
        """)
        
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message:
                with st.expander("View Sources"):
                    for source in message["sources"]:
                        st.markdown(f"""
                        **Source:** {source['metadata']['source']}  
                        **Similarity:** {source['similarity']:.2f}  
                        **Content:** {source['content']}
                        """)

    if prompt := st.chat_input("Ask a question about the blog content..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                query_engine = get_query_engine()
                response = query_engine.get_answer(prompt)
                
                st.markdown(response["answer"])
                
                if response["context_docs"]:
                    with st.expander("View Sources"):
                        for doc in response["context_docs"]:
                            st.markdown(f"""
                            **Source:** {doc['metadata']['source']}  
                            **Similarity:** {doc['similarity']:.2f}  
                            **Content:** {doc['content']}
                            """)
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response["answer"],
                    "sources": response["context_docs"] if response["context_docs"] else []
                })

if __name__ == "__main__":
    main()