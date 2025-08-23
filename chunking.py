''' This file is create by Alaina Srivastav; revised by Yushu Huang
for joint DMBI project of Cindy Lam, Alaina Srivastav and Yushu Huang.
Last revision 08/21/2025. 
Purpose of this file: to handle data chunking. 

Input Data: cleaned papers, presumably in text format. 
Chunking: employ langchain text splitter to chunk the documents.'''


import os
import csv
import pickle
from pathlib import Path
from collections import defaultdict


from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


class chunking():
    def __init__(self, output_folder):
        self.output_folder = output_folder

    def make_chunks(self, docs):
        separators = ['\n\n', ' ', '']
        text_splitter = RecursiveCharacterTextSplitter(separators=separators, 
                                                keep_separator=False, chunk_size=500, chunk_overlap=100)
        chunks = text_splitter.split_documents(docs)
        return chunks
    def save_chunks_to_text(self, docs):
        output_folder = self.output_folder
        # Ensure the output directory exists
        os.makedirs(output_folder, exist_ok=True)

        # Group documents by metadata
        grouped_documents = defaultdict(list)
        for doc in docs:
            metadata = doc.metadata['source'].rsplit('/', 1)[-1]       # define metadata = the file name
            grouped_documents[metadata].append(doc.page_content)

        # Write grouped documents to separate files
        for metadata, contents in grouped_documents.items():
            file_path = os.path.join(output_folder, metadata)
            with open(file_path, 'w', encoding='utf-8') as file:
                for content in contents:
                    file.write(content + '\n')
            print(f"Documents saved with metadata '{metadata}'")

    def save_chunk_for_later_loading(self, docs):

        output_folder = self.output_folder
        # Ensure the output directory exists
        os.makedirs(output_folder, exist_ok=True)
        # Save the Document object to a file
        
        filepath = os.path.join(output_folder, 'document.pkl')
        with open(filepath, 'wb') as f:
            pickle.dump(docs, f)
            print('chunks saved to document.pkl')

    def load_chunks(self):
        load_folder = self.output_folder
        # filepath = os.path.join(load_folder + 'document.pkl')
        filepath = os.path.join(load_folder, 'document.pkl')
    
        if not os.path.exists(filepath):
            print(f"File {filepath} does not exist.")

        with open(filepath, 'rb') as f:
            loaded_doc = pickle.load(f)
        return loaded_doc




def main(input_folder):
    path = os.getcwd()
    data_subpath = '/chunks/'
    output_folder = os.path.join(path + data_subpath)
    if not Path(output_folder).exists():
        os.makedirs(output_folder)
    chunk = chunking(output_folder)

    loader = DirectoryLoader(path=input_folder, glob="**/*.txt", show_progress=True, loader_cls=TextLoader)
    documents = loader.load()

    '''chunking'''
    chunks = chunk.make_chunks(documents)

    '''saving'''
    # chunk.save_chunks_to_text(docs=chunks)
    chunk.save_chunk_for_later_loading(docs=chunks)
    chunks = chunk.load_chunks()
    print(len(chunks))


if __name__ == "__main__":
    path = os.getcwd()
    data_subpath = '/cleaned_papers/'
    input_folder = os.path.join(path + data_subpath)
    if not Path(input_folder).exists():
        raise FileNotFoundError(f"Folder '{input_folder}' was not found")
    main(input_folder)