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
        ''' a prompt TBD. input: list of factors. output: find in context their relation, if the context mentioned. '''
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
                    "You excel at consolidating semantically similar information and producing clear, concise, non-redundant summaries. "
                    "Always aim to merge factors with similar meanings."
                ),
                "role": "user",
                "content": (
                    "Task: Summarize and consolidate key factors from academic context paragraphs.\n\n"
                    "Context provided:\n"
                    f"- A definition: {phenomenon_definition}\n"
                    "- A set of context paragraphs. Each paragraph includes a quotation, a citation (author, year), and sometimes an explanation (which starts with ': ').\n\n"
                    "Instructions:\n"
                    "1. Read and understand the provided definition and all context paragraphs below:\n"
                    f"{chunk}\n\n"
                    f"2. Identify all factors that contribute to {outcome_definition['phenomenon']}. Merge factors that have similar meanings, even if expressed differently, into a single, clearly worded factor.\n"
                    "3. For each consolidated factor, count the number of unique citations (author, year) that reference that factor. If a citation repeats for the same factor, count it only once.\n"
                    "4. Rank the consolidated factors from most to least cited (by unique citation count).\n"
                    "5. Assign a frequency label to each factor:\n"
                    "   - 'high frequency' for the top 1-2 most cited factors\n"
                    "   - 'common' for factors in the middle\n"
                    "   - 'less common' for the least cited factors\n"
                    "6. Output a numbered list. For each factor, include:\n"
                    "   - A concise summary of the consolidated factor\n"
                    "   - The number of unique citations\n"
                    "   - The frequency label\n\n"
                    "Example output:\n"
                    "1. [Summary of consolidated factor 1]\n"
                    "   - Citations: N\n"
                    "   - Frequency: high frequency\n"
                    "2. [Summary of consolidated factor 2]\n"
                    "   - Citations: M\n"
                    "   - Frequency: common\n"
                    "3. [Summary of consolidated factor 3]\n"
                    "   - Citations: X\n"
                    "   - Frequency: less common\n\n"
                    "Guidelines:\n"
                    "- Merge similar factors; avoid redundancies.\n"
                    "- Make each summary clear and unique.\n"
                    )
            }

        #     {"role": "system", 
        #     "content": (
        #         "You are an expert assistant in academic text analysis and synthesis. "
        #         "You excel at consolidating semantically similar information and producing clear, concise summaries."
        #     )},
        #     {"role": "user", "content": (
        #         f"You are given a set of context paragraphs, each containing a quotation, its citation in the format (author, year), and optionally an explanation (starting with ": ").\n"
        #         f"Your task:\n"
        #         f"1. Carefully read and understand the definition of this terminology: {phenomenon_definition} and all context paragraphs.\n"
        #         f"2. Here is the context: \n{chunk}\n\n"
        #         f"2. Identify all factors that contribute to {outcome_definition['phenomenon']}, even if they are described using different words or phrases. Pay careful attention to similarities in meaning and intent, and consolidate such factors into a single, clearly worded factor.\n"
        #         f"3. For each consolidated factor, count the number of unique citations (author, year) that reference this factor (count each citation only once per factor, even if it appears in multiple paragraphs).\n"
        #         f"4. Rank the consolidated factors from most to least frequently cited, according to the number of unique citations.\n"
        #         f"5. Assign a frequency label to each factor:\n"
        #         f"- 'high frequency' for the top 1-2 most cited factors,\n"
        #         f"- 'common' for factors in the middle range,\n"
        #         "- 'less common' for the least cited factors.\n\n"
        #         "6. Output a numbered list where each entry includes:\n"
        #         "- A concise summary of the consolidated factor,\n"
        #         "- The number of unique citations,\n"
        #         "- The frequency label.\n"
        #         "\n Example output:\n\n"
        #         "1. [Summary of consolidated factor 1]\n"
        #         "- Citations: N\n"
        #         "- Frequency: high frequency\n"
        #         "2. [Summary of consolidated factor 2]\n"
        #         "- Citations: M\n"
        #         "- Frequency: common\n"
        #         "3. [Summary of consolidated factor 3]\n"
        #         "- Citations: X\n"
        #         "- Frequency: less common\n\n"
        #         "Please ensure that factors with similar meanings are merged, and avoid listing redundant or overlapping factors. Provide clear and distinct summaries for each consolidated factor."
        #     )
        # }
            
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
            response = client.chat.completions.create(model="gpt-4o",
            messages=message,
            max_tokens=max_tokens,
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


