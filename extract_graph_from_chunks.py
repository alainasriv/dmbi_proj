''' This is a temporary python file created by Yushu.
To demonstrate the idea of using LLMGraphTransformer to create a graph for neo4j to handle. 
Alaina: please merge it to bulk_chunk.py by modifying the extract_graph_from_chunks function there, with potentially necessary tweaks. '''


import os
import getpass
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_community.graphs import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.documents import Document

''' initialize a cloud instance of graph for neo4j. '''
graph = Neo4jGraph(
    url="bolt://54.87.130.140:7687",
    username="neo4j",
    password="cables-anchors-directories",
    refresh_schema=False
)


def extract_graph_from_chunks(chunks, llm):
    ''' llm handles the chunks using LLMGraphTransformer to generate graph
    reference link: https://medium.com/data-science/building-knowledge-graphs-with-llm-graph-transformer-a91045c49b59 
    code reference link: https://github.com/tomasonjo/blogs/blob/master/llm/llm_graph_transformer_in_depth.ipynb 
    Input:  chunked documents. class: Document
            llm: eg. ChatOpenAI(model="gpt-4o")
    Output: graph document, to be handled by neo4j. '''

    transformer = LLMGraphTransformer(llm=llm)

    for i, chunk in enumerate(chunks):
        print(f"Extracting from chunk {i+1}/{len(chunks)}")
        try:
            doc = Document(page_content=chunk)
            graph_documents = transformer.convert_to_graph_documents([doc])
            graph.add_graph_documents(graph_documents)
                
        except Exception as e:
            print(f"Error processing chunk {i+1}: {e}")
        time.sleep(1)

    return graph
        