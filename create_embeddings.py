import os
from typing import List, Dict
import chromadb
from sentence_transformers import SentenceTransformer
import glob

class BlogEmbeddings:
    def __init__(self, data_folder: str = "raw_data", db_path: str = "./chroma_db"):
        self.data_folder = data_folder
        self.db_path = db_path
        self.model = SentenceTransformer('all-mpnet-base-v2')
        self.client_chroma = chromadb.PersistentClient(path=db_path)
        self.collection = self.client_chroma.get_or_create_collection(
            name="blog_embeddings",
            metadata={"description": "Blog post embeddings for RAG"}
        )
        
    def read_markdown_file(self, file_path: str) -> str:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
        
        # Split text into paragraphs based on double line breaks
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for para in paragraphs:
            para_length = len(para)
            
            if current_length + para_length > chunk_size and current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                
                if len(current_chunk) > 0 and len(current_chunk[-1]) < overlap:
                    current_chunk = [current_chunk[-1]]
                    current_length = len(current_chunk[-1])
                else:
                    current_chunk = []
                    current_length = 0
            
            current_chunk.append(para)
            current_length += para_length + 2  
            
            if current_length >= chunk_size/2 and para.endswith(('.', '!', '?')):
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = []
                current_length = 0
        
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        
        final_chunks = []
        min_chunk_size = 200  # Minimum characters for a chunk
        
        for i, chunk in enumerate(chunks):
            if len(chunk) < min_chunk_size and i > 0:
                final_chunks[-1] = final_chunks[-1] + '\n\n' + chunk
            else:
                final_chunks.append(chunk)
        
        return final_chunks
    
    def process_blogs(self) -> Dict[str, List[str]]:
        blog_data = {}
        
        blog_files = glob.glob(os.path.join(self.data_folder, "blog_*.md"))
        
        for file_path in blog_files:
            file_name = os.path.basename(file_path)
            content = self.read_markdown_file(file_path)
            chunks = self.chunk_text(content)
            blog_data[file_name] = chunks
        
        return blog_data
    
    def create_embeddings(self) -> None:
        blog_data = self.process_blogs()

        
        for file_name, chunks in blog_data.items():
            # Generate IDs for each chunk
            ids = [f"{file_name}_chunk_{i}" for i in range(len(chunks))]
            
            embeddings = self.model.encode(chunks)
            
            self.collection.add(
                documents=chunks,
                embeddings=embeddings.tolist(),
                ids=ids,
                metadatas=[{
                    "source": file_name, 
                    "chunk_index": i,
                    "chunk_size": len(chunk)
                } for i, chunk in enumerate(chunks)]
            )
        
        print(f"Successfully processed and stored embeddings for {len(blog_data)} blogs")
        print(f"Total number of chunks stored: {self.collection.count()}")

def main():
    embedder = BlogEmbeddings(data_folder="raw_data", db_path="./chroma_db")
    embedder.create_embeddings()
    print('Embeddings saved successfully!')

if __name__ == "__main__":
    main()