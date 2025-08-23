''' This file is create by Yushu Huang on @date 08/21/2025 
as the main function for joint DMBI project of Cindy Lam, Alaina Srivastav and Yushu Huang.
Last revision 08/21/2025. '''


# %%
import os
import re

from data_process import data_processor
from chunking import chunking
from KGconstruct import extract_graph

from langchain_openai import ChatOpenAI

def data_processing(path, raw_data_folder):

    processor = data_processor(path=path)   # claim the processing object

    loader = processor._extract_text_from_pdf       
    papers = processor.load_all_papers(folder=raw_data_folder, loader=loader)        # load papers in from pdf
    sections = processor.extract_section(papers)              # automated cleaning.
    processor.save_to_text(sections, os.path.join(path + '/cleaned_papers/'))       # save cleaning results as text file. 
    return sections


def main():
    # path
    path = os.getcwd()
    data_subpath = 'pdfs'
    chunk_subpath = 'chunks'
    graph_subpath = 'graph'

    chunk_folder = os.path.join(path, chunk_subpath)
    graph_folder = os.path.join(path, graph_subpath)

    os.makedirs(os.path.join(path, data_subpath), exist_ok=True)
    os.makedirs(chunk_folder, exist_ok=True)
    os.makedirs(graph_folder, exist_ok=True)

    # # preprocess papers
    # papers = data_processing(path=path, raw_data_folder=data_subpath)

    # chunking
    chunk = chunking(chunk_folder)
    chunks = chunk.make_chunks(papers)
    chunk.save_chunks_to_text(docs=chunks)
    chunk.save_chunk_for_later_loading(docs=chunks)
    
    # # graph making
    # llm = ChatOpenAI(model="gpt-4o", temperature=0)
    # graph_extractor = extract_graph()
    # grouped_documents = {}

    # for doc in chunks:
    #     meta = doc.metadata['source']
    #     if meta not in grouped_documents:
    #         grouped_documents[meta] = []
    #     grouped_documents[meta].append(doc)

    # for name in grouped_documents.keys():
    #     # Extract knowledge graph from chunks

    #     kg = graph_extractor.extract_graph_from_chunks(chunk_list=grouped_documents[name], llm=llm)
    #     base_name = os.path.basename(name['source'])
    #     kg_name, _ = os.path.splitext(base_name)
    #     kg_path = os.path.join(graph_folder,kg_name)
    #     with open(kg_path, "w", encoding="utf-8") as f:
    #         json.dump(kg, f, indent=2)
    #     print(f"Saved KG JSON: {kg_path}")


main()
    
