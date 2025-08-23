''' This file is create by Alaina Srivastav; revised by Yushu Huang
for joint DMBI project of Cindy Lam, Alaina Srivastav and Yushu Huang.
Last revision 08/21/2025. 
Purpose of this file: to handle data chunking. 

Input Data: cleaned papers, presumably in text format. 
Chunking: employ langchain text splitter to chunk the documents.'''


import os
import json
import time
from PyPDF2 import PdfReader
from fpdf import FPDF
from pathlib import Path

from chunking import chunking

from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer

class extract_graph():
    def __init__(self, graph_path):
        self.graph_path = graph_path

# ---------- LLM Graph Extraction ----------
def extract_graph_from_chunks(chunk_list, llm):
    
    transformer = LLMGraphTransformer(llm=llm)
    all_entities = []
    all_relations = []

    for chunk in chunk_list:
        print(f"Extracting from chunk {i+1}/{len(chunk.page_content)}")
        try:
            graph_documents = transformer.convert_to_graph_documents([doc])
            
            for graph_doc in graph_documents:
                all_entities.extend(graph_doc.nodes)
                all_relations.extend(graph_doc.relationships)
                
        except Exception as e:
            print(f"Error processing chunk {i+1}: {e}")
        time.sleep(1)

    return {
        "entities": [{"id": e.id, "type": e.type, "properties": e.properties} for e in all_entities],
        "relations": [{"source": r.source.id, "target": r.target.id, "type": r.type, "properties": r.properties} for r in all_relations]
    }

# ---------- Main Pipeline ----------
def main():
    path = os.getcwd()
    data_subpath = 'chunks'
    graph_subpath = 'graph'
    chunk_folder = os.path.join(path, data_subpath)
    graph_folder = os.path.join(path, graph_subpath)
    os.makedirs(graph_folder, exist_ok=True)
    if not Path(chunk_folder).exists():
        raise FileNotFoundError(f"Folder '{chunk_folder}' was not found")
    
    chunk = chunking(chunk_folder)
    chunks = chunk.load_chunks()

    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    graph_extractor = extract_graph(graph_folder)

    grouped_documents = {}
    for doc in chunks:
        meta = doc.metadata['source']
        if meta not in grouped_documents:
            grouped_documents[meta] = []
        grouped_documents[meta].append(doc)

    for name in grouped_documents.keys():
        # Extract knowledge graph from chunks

        kg = graph_extractor.extract_graph_from_chunks(chunk_list=grouped_documents[name], llm=llm)
        base_name = os.path.basename(name['source'])
        kg_name, _ = os.path.splitext(base_name)
        kg_path = os.path.join(graph_folder,kg_name)
        with open(kg_path, "w", encoding="utf-8") as f:
            json.dump(kg, f, indent=2)
        print(f"Saved KG JSON: {kg_path}")

if __name__ == "__main__":
    main()