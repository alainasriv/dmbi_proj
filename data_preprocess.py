''' This file is create by Yushu Huang on @date 08/13/2025 
for joint DMBI project of Cindy Lam, Alaina Srivastav and Yushu Huang.
Last revision 08/13/2025. 
Purpose of this file: to handle data cleaning. 

Data: papers, presumably in pdf or text format. 
Cleaning: removing all tables, extracting only parts including and after Results section upto before Reference section.'''

# %%
import os
import re
import pdfplumber
from PyPDF2 import PdfReader
# %%
path = os.getcwd()
data_subpath = '/pdfs/'

# %% load papers

def _extract_non_table_text(pdf_path):
    ''' load papers from pdf, removing tables'''
    try:
        with pdfplumber.open(pdf_path) as pdf:
            all_text = []
            for page in pdf.pages:
                # Get text
                text = page.extract_text()
                # Get tables
                tables = page.extract_tables()

                # Convert tables into text form (if needed) and exclude from the main text
                tables_text = '\n'.join(['\n'.join([' '.join(row) for row in table]) for table in tables if table])
                # Remove tables from main text
                non_table_text = text.replace(tables_text, "")
                all_text.append(non_table_text)

            return '\n'.join(all_text)
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        return ""

def _extract_text_from_pdf(pdf_path):
    ''' load papers from pdf, not removing tables'''
    try:
        with open(pdf_path, "rb") as f:
            reader = PdfReader(f)
            text = ""
            for i, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + " "
                except Exception as e:
                    print(f"Error reading page {i+1} in {pdf_path}: {e}")
            return text.strip()
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        return ""

def _extract_text_from_txt(txt_path):
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"Error reading {txt_path}: {e}")
        return ""
    
# %% 
def load_all_papers(directory, loader):
    papers = {}
    for filename in os.listdir(directory):
        path = os.path.join(directory, filename)
        if filename.endswith(".pdf"):
            text = loader(path)
            if text:
                papers[filename] = text
        elif filename.endswith(".txt"):
            text = loader(path)
            if text:
                papers[filename] = text
    print(f"Successfully loaded {len(papers)} papers")
    return papers

# %% section extraction
def _peel_paper_with_regex(paper_name, text):
    start_regex = r'\nresults?(?: of .*)?\n'
    end_regex = r'\nreference?(?: of .*)?'

    temp = len(re.findall(start_regex, text))
    if temp == 0:
        raise ValueError(paper_name + "No starting point cannot be found for section extraction")
    elif temp >1:
        raise ValueError(paper_name + "Multiple starting point cannot be found for section extraction")
    
    temp = len(re.findall(end_regex, text))
    if temp == 0:
        raise ValueError(paper_name + "No ending point cannot be found for section extraction")
    elif temp >1:
        raise ValueError(paper_name + "Multiple ending point cannot be found for section extraction")
    # Use the compiled regular expression to find text between start and end markers
    pattern = re.compile(r'(?i)(' + start_regex + r'(.*?)' + end_regex +r')' , re.DOTALL)
    matches = pattern.findall(text)
    return matches
# %% main
def main():
    input_dir = os.path.join(path+ data_subpath)
    loader = _extract_text_from_pdf
    papers = load_all_papers(input_dir, loader=loader)
    return papers




# %%
if __name__ == "__main__":
    papers = main()
# %%
