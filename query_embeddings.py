from typing import List, Dict
import chromadb
from groq import Groq
from sentence_transformers import SentenceTransformer

class BlogQueryEngine:
    def __init__(
        self, 
        groq_api_key: str,
        db_path: str = "./chroma_db",
        model_name: str = "mixtral-8x7b-32768",
        collection_name: str = "blog_embeddings",
        similarity_threshold: float = 0.3
    ):
        self.client_chroma = chromadb.PersistentClient(path=db_path)
        self.collection = self.client_chroma.get_collection(collection_name)
        self.groq_client = Groq(api_key=groq_api_key)
        self.model_name = model_name
        self.embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        self.similarity_threshold = similarity_threshold
        
    def get_relevant_docs(self, query: str, n_results: int = 3) -> List[Dict]:
        query_embedding = self.embedding_model.encode([query])
        
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        
        relevant_docs = []
        for doc, metadata, distance in zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        ):
            similarity = 1 - distance
            if similarity >= self.similarity_threshold:
                relevant_docs.append({
                    "content": doc,
                    "metadata": metadata,
                    "similarity": similarity
                })
        
        return relevant_docs

    def generate_prompt(self, query: str, relevant_docs: List[Dict]) -> str:
        if relevant_docs:
            context_parts = []
            for i, doc in enumerate(relevant_docs, 1):
                source = doc['metadata']['source']
                content = doc['content']
                context_parts.append(f"[{source}]: {content}")
            
            context_str = "\n".join(context_parts)
            
            return f"""You are a helpful AI assistant. Use the following blog content to answer the question. Use most of the context from blog content, with small meaningful addition of general knowledge.

Blog Content:
{context_str}

Question: {query}

Answer:"""
        else:
            return f"""No relevant blog content found. Please answer this question based on your general knowledge.

Question: {query}

Answer:"""

    def get_answer(self, query: str) -> Dict:
        relevant_docs = self.get_relevant_docs(query)
        prompt = self.generate_prompt(query, relevant_docs)
        
        completion = self.groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant. Prioritize using the provided blog content when answering questions. Cite the blog source when using blog content."},
                {"role": "user", "content": prompt}
            ],
            model=self.model_name,
            temperature=0.3,
            max_tokens=1000
        )
        
        response = {
            "answer": completion.choices[0].message.content,
            "used_context": len(relevant_docs) > 0,
            "context_docs": relevant_docs if relevant_docs else None
        }
        
        return response