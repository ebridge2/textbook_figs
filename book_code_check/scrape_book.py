import os
import re
import json
import sys
import argparse
from nbformat import v4 as nbf

def extract_python_code(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    pattern = r'\\begin{lstlisting}\[style=python\](.*?)\\end{lstlisting}'
    matches = re.findall(pattern, content, re.DOTALL)
    
    return [match.strip() for match in matches]

def create_notebook(code_snippets, output_file, section):
    nb = nbf.new_notebook()
    
    # Add the text cell at the beginning
    text_content = f"({section}:code_repr)=\n# Code Reproducibility"
    nb['cells'].append(nbf.new_markdown_cell(text_content))
    
    for snippet in code_snippets:
        nb['cells'].append(nbf.new_code_cell(snippet))
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2)

def get_input_order(main_file):
    with open(main_file, 'r', encoding='utf-8') as file:
        content = file.read()
    
    pattern = r'\\input{(.*?)}'
    matches = re.findall(pattern, content)
    
    return [os.path.basename(match) for match in matches]

def process_directory(dir_path, main_file_name):
    all_code = []
    main_file_path = os.path.join(dir_path, main_file_name)
    
    if not os.path.exists(main_file_path):
        print(f"Main file not found: {main_file_path}")
        return all_code

    input_order = get_input_order(main_file_path)
    
    # Process the main file first
    all_code.extend(extract_python_code(main_file_path))
    
    # Process input files in order
    for input_file in input_order:
        file_path = os.path.join(dir_path, f"{input_file}.tex")
        if os.path.exists(file_path):
            all_code.extend(extract_python_code(file_path))
        else:
            print(f"Input file not found: {file_path}")
    
    return all_code

def main(base_dir, output_dir):
    top_level_dirs = ['appendix', 'applications', 'foundations', 'representations']
    
    for top_dir in top_level_dirs:
        top_dir_path = os.path.join(base_dir, top_dir)
        if os.path.isdir(top_dir_path):
            sub_dirs = [d for d in os.listdir(top_dir_path) 
                        if os.path.isdir(os.path.join(top_dir_path, d))]
            
            for sub_dir in sub_dirs:
                dir_path = os.path.join(top_dir_path, sub_dir)
                main_file_name = next((f for f in os.listdir(dir_path) if f.startswith(('ch', 'app')) and f.endswith('.tex')), None)
                
                if main_file_name:
                    all_code = process_directory(dir_path, main_file_name)
                    
                    if all_code:
                        output_file = os.path.join(output_dir, f'{top_dir}_{sub_dir}_notebook.ipynb')
                        section = sub_dir  # Keep the 'ch' or 'app' prefix
                        create_notebook(all_code, output_file, section)
                        print(f'Created notebook: {output_file}')
                    else:
                        print(f'No Python code found in: {dir_path}')
                else:
                    print(f"No main chapter file found in: {dir_path}")
        else:
            print(f"Directory not found: {top_dir_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert LaTeX Python code to Jupyter notebooks.')
    parser.add_argument('base_dir', type=str, help='Base directory of the LaTeX project')
    parser.add_argument('--output_dir', type=str, default='output', help='Output directory for Jupyter notebooks (default: ./output)')
    args = parser.parse_args()

    if not os.path.isdir(args.base_dir):
        print(f"Error: The specified base directory does not exist: {args.base_dir}")
        sys.exit(1)

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    main(args.base_dir, output_dir)
