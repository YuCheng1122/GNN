import json
from turtle import dot
import networkx as nx
from pathlib import Path
from typing import Dict, List, Tuple, Generator
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import pickle

def read_csv(csv_file_path: str | Path) -> List[List[str]]:
    df = pd.read_csv(csv_file_path)
    file_names = df['file_name'].tolist()
    return file_names

def clean_data(json_data, G_raw: nx.DiGraph) -> nx.DiGraph:
    G = nx.DiGraph()
    for node in G_raw.nodes():
        addr = str(node)
        func = json_data.get(addr)
        if not func:
            continue

        instructions = func.get("instructions", [])
        pcode_list = []

        for instr in instructions:
            if isinstance(instr, dict):
                operation = instr.get("opcode")
                if isinstance(operation, str):  
                    pcode_list.append(operation)

        if pcode_list:
            G.add_node(addr, pcode=pcode_list)
    for src, dst in G_raw.edges():
        src, dst = str(src), str(dst)
        if G.has_node(src) and G.has_node(dst):
            G.add_edge(src, dst)

    return G


def process_single_file_data(file_info, output_base_path):
    json_path, dot_path, file_name = file_info
    
    try:
        with open(json_path, 'r') as f:
            json_data = json.load(f)
        G_raw = nx.drawing.nx_pydot.read_dot(dot_path)
        G = clean_data(json_data, G_raw)
        
        # Prepare output path
        prefix = file_name[:2]
        output_dir = output_base_path / prefix
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{file_name}.gpickle"
        
        try:
            with open(output_file, 'wb') as f:
                pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print(f"Error saving graph to {output_file}: {str(e)}")
            raise

        # Clear variables to free memory
        del json_data, G_raw, G
        
        return f"Successfully processed {file_name}"
    except Exception as e:
        return f"Error processing {file_name}: {str(e)}"


def process_files(csv_file_path: str | Path, root_dir: str | Path, output_base_dir: str | Path, num_processes=None):
    root_path = Path(root_dir)
    output_base_path = Path(output_base_dir)

    
    # Collect file paths instead of loading data
    file_info_list = []
    file_names = read_csv(csv_file_path)
    
    for file_name in tqdm(file_names, desc="Collecting file paths"):
        json_path = root_path / file_name / f"{file_name}.json"
        dot_path = root_path / file_name / f"{file_name}.dot"
        
        if json_path.exists() and dot_path.exists():
            file_info_list.append((json_path, dot_path, file_name))
        else:
            missing = []
            if not json_path.exists():
                missing.append("JSON")
            if not dot_path.exists():
                missing.append("DOT")
            print(f"Missing {', '.join(missing)} file(s) for: {file_name}")
    
    # Use multiprocessing to process files in parallel
    if num_processes is None:
        num_processes = mp.cpu_count()
    
    print(f"Processing {len(file_info_list)} files using {num_processes} processes...")
    
    # Create a partial function with fixed output_base_path
    process_func = partial(process_single_file_data, output_base_path=output_base_path)
    
    # Use multiprocessing pool to process files
    with mp.Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap(process_func, file_info_list),
            total=len(file_info_list),
            desc="Processing files"
        ))
    
    success_results = [r for r in results if not r.startswith("Error")]
    failure_results = [r for r in results if r.startswith("Error")]
    
    successes = len(success_results)
    failures = len(failure_results)
    
    print(f"Processing complete: {successes} files processed successfully, {failures} failures")

def main():
    csv_file_path = "/home/tommy/Projects/cross-architecture/Experiment3.1/dataset/cleaned_20250509_test_600.csv" 
    root_dir = "/home/tommy/Projects/cross-architecture/reverse/output_new/results" 
    output_base_dir = "/home/tommy/Projects/cross-architecture/Gpickle/20250509_new_test_600_token"
    process_files(csv_file_path, root_dir, output_base_dir)
    
    print("Processing complete. All gpickle files have been saved.")

if __name__ == "__main__":
    main()