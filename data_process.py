''' This file is create by Yushu Huang on @date 08/13/2025 
for joint DMBI project of Cindy Lam, Alaina Srivastav and Yushu Huang.
Last revision 08/21/2025. 
Purpose of this file: to handle data cleaning. 

Input Data: papers, presumably in pdf or text format. 
Cleaning: removing all tables, extracting only parts including and after Results section upto before Reference section.'''

# %%
import os
import re
from pathlib import Path
import pdfplumber
from PyPDF2 import PdfReader

# %% 
class data_processor():
    def __init__(self, path):
        self.path = path

    # %% load papers

    def _extract_text_from_pdf_notable(self, pdf_path):
        ''' load papers from pdf, removing tables'''

        try:
            with pdfplumber.open(pdf_path) as pdf:
                all_text = []
                for i, page in enumerate(pdf.pages):
                    try:
                        # Get text
                        text = page.extract_text()
                        # Get tables
                        tables = page.extract_tables()

                        # Convert tables into text form (if needed) and exclude from the main text
                        tables_text = '\n'.join(['\n'.join([' '.join(row) for row in table]) for table in tables if table])
                        # Remove tables from main text
                        non_table_text = text.replace(tables_text, "")
                        all_text.append(non_table_text)
                    except Exception as e:
                        print(f"Error reading page {i+1} in {pdf_path}: {e}")

                return '\n'.join(all_text)
        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")
            return ""

    def _extract_text_from_pdf(self, pdf_path):
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

    def _extract_text_from_txt(self, txt_path):
        
        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            print(f"Error reading {txt_path}: {e}")
            return ""
        
    # %% 
    def load_all_papers(self, folder, loader):
        path = self.path
        directory = os.path.join(path + folder)
        if not Path(directory).exists():
            raise FileNotFoundError(f"Folder '{directory}' was not found")
        files = os.listdir(directory)
        if not files:
            raise FileNotFoundError(f"No files found in the folder '{directory}'.")

        papers = {}
        for filename in files:
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
    def _peel_paper_with_regex(self, paper_name, text):

        # Use the regular expression to find result section as start and reference section as end. 
        start_regex = r'(?i)\n(?:r\s*e\s*s\s*u\s*l\s*t\s*s?)\n'
        end_regex = r'(?i)\n(?:r\s*e\s*f\s*e\s*r\s*e\s*n\s*c\s*e\s*s?)\n'

        temp = len(re.findall(start_regex, text))
        if temp == 0:
            raise ValueError(paper_name + " No starting point found for section extraction")
        elif temp >1:
            raise ValueError(paper_name + " Multiple starting point found for section extraction")
        
        temp = len(re.findall(end_regex, text))
        if temp == 0:
            raise ValueError(paper_name + " No ending point found for section extraction")
        elif temp >1:
            raise ValueError(paper_name + " Multiple ending point found for section extraction")
        
        # Use the compiled regular expression to find text between start and end markers
        start_regex = r'\n(?:r\s*e\s*s\s*u\s*l\s*t\s*s?)\n'
        end_regex = r'\n(?:r\s*e\s*f\s*e\s*r\s*e\s*n\s*c\s*e\s*s?)\n'

        pattern = re.compile(r'(?i)(' + start_regex + r'(.*?)' + end_regex +r')' , re.DOTALL)
        matches = pattern.findall(text)
        return matches[0][0]

    # %% 

    def extract_section(self, papers):

        extracted_papers = {}
        for name, text in papers.items():
            try:
                extracted_papers[name] = self._peel_paper_with_regex(name, text)
                # extracted_papers = extracted_papers + extract_section(name, text)
            except ValueError as e:
                print(f"Error: {e}")
                extracted_papers[name] = text
        return extracted_papers

    # %% save results to text
    def save_to_text(self, papers, folder):

        os.makedirs(folder, exist_ok=True)
        for name, text in papers.items():
            if name.endswith(".pdf"):
                name = name[:-4]
            path = os.path.join(folder + name + '.txt')
            with open(path, "w") as f:
                f.write(text)
                print(f"Saving completed for file '{name}. ")
# %% main
def main():

    path = os.getcwd()
    data_subpath = '/pdfs/'
    processor = data_processor(path=path)

    # input_dir = os.path.join(path+ data_subpath)
    
    loader = processor._extract_text_from_pdf
    papers = processor.load_all_papers(folder=data_subpath, loader=loader)

    sections = processor.extract_section(papers)
    # for name, text in sections.items():
    #     if name.endswith(".pdf"):
    #         name = name[:-4]
    #     print(os.path.join(path + '/cleaned_papers/' + name + '.txt'))

    processor.save_to_text(sections, os.path.join(path + '/cleaned_papers/'))

    return sections




# %%
if __name__ == "__main__":
    papers = main()
# %%
