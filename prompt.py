''' This file is create by Yushu Huang
for joint DMBI project of Cindy Lam, Alaina Srivastav and Yushu Huang.
Last revision 08/28/2025. 
Purpose of this file: to prompt for factor -> outcome summarization. 

Input Data: chunks as document.pkl. 
Chained prompt engineering'''

# %%
import os
import re
import json

from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_openai import ChatOpenAI
from langchain_core.documents.base import Document

from chunking import chunking


class prompt():
    def __init__(self, chunked_docs, **kwargs):
        self.chunked_docs = chunked_docs
        self.kwargs       = kwargs

    def _message_find_factors_to_result(self, chunk):

        # Check if 'outcome_definition' is in kwargs, else raise an error
        if 'outcome_definition' in self.kwargs:
            outcome_definition = self.kwargs['outcome_definition']
        else:
            raise ValueError("The keyword argument 'outcome_definition' is required.")


        phenomenon_definition = "\n\n".join(
            f"- **{item['phenomenon']}**: {item['definition']}" for item in outcome_definition
        )
        response_format = "\n".join(
            f"The factors contributing to {item['phenomenon']} are: (list all factors contributing to {item['phenomenon']} only)"
            for item in outcome_definition
        )
        messages = [
            {"role": "system", 
            "content": (
                "You are an assistant skilled at extracting and summarizing key factors from text. "
                "While you do not possess specific domain expertise, you excel in information retrieval from provided content."
            )},
            {"role": "user", "content": (
                f"You are provided with context from a research study on digital mindfulness-based interventions. "
                f"Your task is to identify factors in the context that contributes to each of the following phenomena individually:\n\n"
                f"{phenomenon_definition}\n\n"
                f"Here is the context:\n{chunk}\n\n"
                "Your response should follow this format:\n"
                f"{response_format}\n\n"
                "For each phenomenon, identify all relevant factors from the context." 
                "For each factor, quote the specific part of the context that describe how this factor contribute to this phenomenon.  "
                # "Citations are provided within the context, enclosed in parentheses and formatted like 'Author et al., Year'. "
                "Citations are provided within the context, enclosed in brackets or parenthesis and formatted as an integer or integers separated by comma. "
                "If there are multiple citations, they are separated by a semicolon ';'. Please copy these citations directly into your response. "
                "If no factors are evident, respond with 'There are no related factors in this chunk.'"

            )}
        ]
        return messages
        
    def _message_consolidate_factors(self, chunk):
        ''' a prompt integrating raw factors. '''
        # Check if 'outcome_definition' is in kwargs, else raise an error
        if 'outcome_definition' in self.kwargs:
            outcome_definition = self.kwargs['outcome_definition']
        else:
            raise ValueError("The keyword argument 'outcome_definition' is required.")
        phenomenon_definition = "\n\n".join(
            f"- **{outcome_definition['phenomenon']}**: {outcome_definition['definition']}"
        )
        message = [
            {
                "role": "system",
                "content": (
                    "You are an expert assistant in academic text analysis and synthesis. "
                    "You specialize in consolidating semantically similar information and producing clear, concise, non-redundant summaries."
                )
            },
            {
                "role": "user",
                "content": (
                    "Task: Summarize and consolidate key factors from provided academic context paragraphs.\n\n"
                    "Context:\n"
                    f"- Definition: {phenomenon_definition}\n"
                    "- The following are context paragraphs. Each includes a quotation, a citation (author, year), and sometimes an explanation (prefixed with ': ').\n\n"
                    f"{chunk}\n\n"
                    "Instructions:\n"
                    f"1. Identify all unique factors that contribute to {outcome_definition['phenomenon']}.\n"
                    "2. Merge factors with similar meanings into a single, clearly named factor. The factor's name should make its relationship to the phenomenon obvious.\n"
                    "3. For each factor, provide a concise explanation of its contribution to the phenomenon.\n"
                    "4. List all unique citations (author, year) supporting each factor, and provide the count of unique citations. Do not count duplicates.\n"
                    # "5. For each factor, describe any relationships it has with other factors you generate, including:\n"
                    # "   - Factor: the other factor that this factor is related to. \n"
                    # "   - Conditions: Specify 'when/for whom/under what implementation' the above relationship holds. Include all contradictory or alternative conditions if present.\n"
                    # "   - Evidence_span: Provide the minimal text snippet (or pointer) justifying the above conditions.\n"
                    f"5. Include the study ID from the metadata: {chunk.metadata}\n"
                    "6. Rank the consolidated factors from most to least cited (by unique citation count).\n"
                    "7. Output as a numbered list. For each factor, include:\n"
                    "   - A clear summary of the consolidated factor\n"
                    "   - A concise explanation/description\n"
                    "   - Number of unique citations and their list\n"
                    # "   - Any relevant conditions and evidence_spans for relationships with other factors\n"
                    "   - Study_id: {chunk.metadata}\n\n"
                    "Example format:\n"
                    "1. [Consolidated Factor 1]\n"
                    "   - Description: ...\n"
                    "   - Citations: N (list)\n"
                    "   - Study_id: ...\n"
                    "2. [Consolidated Factor 2]\n"
                    "   - Description: ...\n"
                    "   - Citations: M (list)\n"
                    # "   - Condition1: ...\n"
                    # "   - Evidence_span1: ...\n"
                    "   - Study_id: ...\n"
                    "3. [Consolidated Factor 3]\n"
                    "   - Description: ...\n"
                    "   - Citations: X (list)\n"
                    # "   - Condition1: ...\n"
                    # "   - Evidence_span1: ...\n"
                    # "   - Condition2: ...\n"
                    # "   - Evidence_span2: ...\n"
                    "   - Study_id: ...\n\n"
                    "Guidelines:\n"
                    "- Always merge similar factors and avoid redundancy.\n"
                    "- Ensure each summary is clear, unique, and directly related to the phenomenon.\n"
                )
            }
        ]

        return message
    
    def _message_final_factors(self, chunk):
        ''' a prompt prodiucing final combined factors. '''
        # Check if 'outcome_definition' is in kwargs, else raise an error
        if 'outcome_definition' in self.kwargs:
            outcome_definition = self.kwargs['outcome_definition']
        else:
            raise ValueError("The keyword argument 'outcome_definition' is required.")
        phenomenon_definition = "\n\n".join(
            f"- **{outcome_definition['phenomenon']}**: {outcome_definition['definition']}"
        )
        message = [
            {
                "role": "system",
                "content": (
                    "You are an expert in academic text analysis and synthesis. "
                    "You excel at merging semantically similar information and generating clear, concise, non-redundant summaries."
                )
            },
            {
                "role": "user",
                "content": (
                    "Task: Summarize and consolidate key factors from academic context paragraphs.\n\n"
                    "Context:\n"
                    f"- Definition: {phenomenon_definition}\n"
                    "- Below is a set of enumerated factors. Each includes a description, summary explanation, citation count and list, relation to other factors (with conditions and evidence_spans), frequency, and study IDs.\n\n"
                    f"{chunk}\n\n"
                    "Instructions:\n"
                    f"1. Review the definition and all factors provided above.\n"
                    f"2. Consolidate all factors that contribute to {outcome_definition['phenomenon']}. Merge factors with similar meanings into a single, clearly named factor that directly expresses its relation to {outcome_definition['phenomenon']}.\n"
                    "3. For each consolidated factor, provide a clear description and an explanation of how it contributes to the phenomenon.\n"
                    "4. List all unique supporting citations (author, year) for each factor, and state the number of unique citations. Count repeat citations only once.\n"
                    # "5. If there are 'factors', 'conditions' and 'evidence_span' in context, then for each factor, consolidate its relationships to other factors, including:\n"
                    # "   - Factor: the other factor that this factor is related to. \n"
                    # "   - Conditions: Specify 'when/for whom/under what implementation' the above relationship holds. Include all contradictory or alternative conditions if present.\n"
                    # "   - Evidence_span: Provide the minimal text snippet (or pointer) justifying the above conditions.\n"
                    " If there is none, the skip this part. \n"
                    "5. Pool all unique study_ids from the context for each consolidated factor.\n"
                    "6. Rank the factors from most to least cited (by unique citation count).\n"
                    "7. Assign a frequency label to each factor based on citation count and any frequency annotations in the context:\n"
                    "   - 'high frequency' for factors that appear repeatedly or are labeled 'high frequency'\n"
                    "   - 'common' for factors in the middle range\n"
                    "   - 'less common' for rarely seen factors or those labeled 'less common'\n"
                    "8. Format your output as a numbered list. For each factor, include:\n"
                    "   - A concise summary of the factor\n"
                    "   - Description and explanation\n"
                    "   - Number and list of unique citations\n"
                    "   - Frequency label\n"
                    # "   - For each related factor: Condition(s) and Evidence_span(s) supporting the relationship\n"
                    "   - Study_id: pooled unique study_ids\n\n"
                    "Example output:\n"
                    "1. [Consolidated Factor 1]\n"
                    "   - Description: [Explanation]\n"
                    "   - Citations: N ([citation1], [citation2], ...)\n"
                    "   - Frequency: high frequency\n"
                    # "   - Condition1: [Qualifier relating to factor X]\n"
                    # "   - Evidence_span1: [Text snippet for factor X]\n"
                    # "   - Condition2: [Qualifier relating to factor Y]\n"
                    # "   - Evidence_span2: [Text snippet for factor Y]\n"
                    "   - Study_id: [id1, id2, ...]\n"
                    "2. [Consolidated Factor 2]\n"
                    "   - Description: ...\n"
                    "   - Citations: ...\n"
                    "   - Frequency: ...\n"
                    # "   - Condition1: ...\n"
                    # "   - Evidence_span1: ...\n"
                    "   - Study_id: ...\n"
                    "...\n\n"
                    "Guidelines:\n"
                    "- Always merge factors with similar meanings to avoid redundancy.\n"
                    "- Present each summary clearly and uniquely.\n"
                    "- Only count each citation once per factor.\n"
                    "- Consolidate and clarify conditions and evidence_spans from the context for each factor.\n"
                )
            }
        ]


        return message

    def generate_response(self, message_func, include_chunk=False):

        chunks  = self.chunked_docs

        if 'max_tokens' in self.kwargs:
            max_tokens = self.kwargs['max_tokens']
        else:
            max_tokens=1000
        
        results = []

        for i in range(len(chunks)):
            print(f'processing chunk {i}')
            chunk = chunks[i]
            if include_chunk == True:
                if type(chunk) == Document:
                    results.append(chunk.page_content) 
                elif type(chunk) == str:
                    results.append(chunk)
                else:
                    raise TypeError("chunks are either list of Documents or strings")
            message  = message_func(chunk)
            response = client.chat.completions.create(model="gpt-5.2",
            messages=message,
            max_completion_tokens=max_tokens,
            temperature=0)
            results.append(response.choices[0].message.content.strip())
        
        return results

    def write_summaries_to_txt(self, textlist_to_write, output_file):
        '''text_to_write: a list of texts. Each entry is a text generated by analyze_chunk'''
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as outfile:
            for text in textlist_to_write:
                outfile.write(text + "\n\n")
        print("Analyze results saved to text file. ")




#%%

if __name__ == "__main__":
    path = os.getcwd()
    data_subpath = 'work_dir/factors/'
    cleaned_data_folder = os.path.join(path, data_subpath)

    
    loader = DirectoryLoader(path=cleaned_data_folder, glob="**/*.txt", show_progress=True, loader_cls=TextLoader)
    documents = loader.load()

    # Define chunking write-to directory
    chunk_folder = os.path.join(path, data_subpath,'chunks')
    os.makedirs(chunk_folder, exist_ok=True)
    # Chunk
    chunk = chunking(chunk_folder,chunk_size=5000)
    chunks = chunk.make_chunks(documents)

    outcome_filename = 'outcome_definition.json'
    with open(os.path.join(path, outcome_filename)) as outcome_file:
        outcome_definition = json.load(outcome_file)

    # generate_prompt = prompt(chunked_docs = documents, outcome_definition = outcome_definition)
    # message_func = generate_prompt._message_find_factors_to_result
    # results = generate_prompt.generate_response(message_func, include_chunk=True)
    # generate_prompt.write_summaries_to_txt(results, 'response.txt')

    generate_prompt = prompt(chunked_docs=chunks,outcome_definition=outcome_definition[2],max_tokens=5000)
    message_func = generate_prompt._message_consolidate_factors
    results = generate_prompt.generate_response(message_func, include_chunk=True)
    generate_prompt.write_summaries_to_txt(results, os.path.join(path, data_subpath,'response.txt'))


