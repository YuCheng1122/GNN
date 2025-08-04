from typing import List, Dict, Sequence
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import networkx as nx
import numpy as np
from gensim.models import Word2Vec
import pickle

from preprocessing import read_csv, _tokenize_line


def iterate_Gpickle(csv_file_path: str | Path, root_dir: str | Path):
    root_path = Path(root_dir)
    for file_name in tqdm(read_csv(csv_file_path), desc="Processing Gpickle files"):
        prefix = file_name[:2]
        path = root_path / prefix / f"{file_name}.gpickle"
        if path.exists():
            try:
                with open(path, "rb") as fp:
                    G = pickle.load(fp)
                yield path, G 
            except Exception as e:
                tqdm.write(f"[Error] Load Gpickle Failed {path}: {e}")
        else:
            tqdm.write(f"[Warning] File Not Found: {file_name}.gpickle")
           
                
def vectorize_graph_nodes(G: nx.Graph, model: Word2Vec):
    zero_vec = np.zeros(model.vector_size, dtype=float)
    for addr, data in G.nodes(data=True):
        opcodes = data.get("pcode", [])
        vectors = [model.wv[op] for op in opcodes if op in model.wv]
        data["vector"] = np.mean(vectors, axis=0) if vectors else zero_vec
        data.pop("pcode", None)


def save_graph_with_vectors(csv_file_path: str | Path, gpickle_root: str | Path,
                            out_root: str | Path, w2v_model_path: str | Path):
    gpickle_root = Path(gpickle_root)
    out_root = Path(out_root)
    model = Word2Vec.load(str(w2v_model_path))

    for gpath, G in iterate_Gpickle(csv_file_path, gpickle_root):
        vectorize_graph_nodes(G, model)

        out_path = out_root / gpath.relative_to(gpickle_root)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "wb") as fp:
            pickle.dump(G, fp)

        
if __name__ == "__main__":
    CSV_FILE_PATH = "/home/tommy/Projects/cross-architecture/Experiment3.1/dataset/cleaned_20250509_test_600.csv"
    GPICKLE_DIR = "/home/tommy/Projects/cross-architecture/Gpickle/20250509_new_test_600_token"
    VECTOR_DIR = "/home/tommy/Projects/cross-architecture/Vector/20250509_new_test_600_token/data"
    WORD2VEC_MODEL_PATH = "/home/tommy/Projects/cross-architecture/Vector/20250509_new_train_450_token/model/word2vec_20250509_train_450.model"
      
    save_graph_with_vectors(
        CSV_FILE_PATH,
        GPICKLE_DIR,
        VECTOR_DIR,
        WORD2VEC_MODEL_PATH
    )