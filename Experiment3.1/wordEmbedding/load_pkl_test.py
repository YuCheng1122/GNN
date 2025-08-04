from itertools import islice
from pathlib import Path
from multiprocessing import Pool, cpu_count
import pickle
from tqdm import tqdm
from preprocessing import iterate_json_files, _tokenize_line


DATA_DIR = Path("/home/tommy/Projects/cross-architecture/reverse/output_new/results")
TRAIN_CSV_PATH = Path("/home/tommy/Projects/cross-architecture/Experiment3.1/dataset/cleaned_20250509_test_600.csv")
OUTPUT_DIR = Path("/home/tommy/Projects/cross-architecture/Vector/20250509_new_test_600/model")
PICKLE_PATH = OUTPUT_DIR / "sentences_20250509_test_600.pkl"
LOG_PATH = OUTPUT_DIR / "missing_files.log"
BATCH_FILES = 1000

if not OUTPUT_DIR.exists():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def extract_sentences_from_file(file_name_data):
    file_name, pcode_dict = file_name_data
    sentences = []

    try:
        for func_data in pcode_dict.values():
            for instruction in func_data.get("instructions", []):
                op_str = instruction.get("operation")
                if isinstance(op_str, str):
                    tokens = _tokenize_line(op_str)
                    if tokens:
                        sentences.append(tokens)
    except Exception as e:
        print(f"Error processing file {file_name}: {e}")

    return sentences
                                                                     

def corpus_generator(csv_path: Path, root_dir: Path, batch_files:int):
    batch_num = 0
    file_iter = iterate_json_files(csv_path, root_dir)
    batch_paths = []
    
    while True:   
        current_batch = list(islice(file_iter, batch_files))
        if not current_batch:
            break 
        sentences = []
        with Pool(cpu_count()) as pool:
            for sent_list in tqdm(
                pool.imap_unordered(extract_sentences_from_file, current_batch, chunksize=1),
                total=len(current_batch),
                desc=f"Batch {batch_num}"
            ):
                sentences.extend(sent_list)
        out_path = OUTPUT_DIR / f"sentences_batch_{batch_num}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(sentences, f)
        batch_paths.append(out_path)
        del sentences
        del current_batch

        batch_num += 1
        
    print("Merage all batches")
    all_sentences = []
    for path in tqdm(batch_paths, desc="Merging batches"):
        with open(path, "rb") as f:
            all_sentences.extend(pickle.load(f))

    with open(PICKLE_PATH, "wb") as f:
        pickle.dump(all_sentences, f)
        
    for path in batch_paths:
        path.unlink()

if __name__ == "__main__":
    corpus_generator(TRAIN_CSV_PATH, DATA_DIR, BATCH_FILES)
    
    
            
