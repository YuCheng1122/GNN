from cProfile import label
from logging import root
import scipy.sparse as sp
import numpy as np
import scipy.sparse as sp
import networkx as nx
from pathlib import Path
from tqdm import tqdm
import pickle
from typing import Dict, List, Tuple, Generator, Sequence
import pandas as pd
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import torch


def iterate_Gpickle(
    csv_file_path: str | Path,
    root_dir: str | Path
) -> Generator[Tuple[Path, nx.DiGraph, Dict[str, Sequence[str]]], None, None]:
    """
    Iterate through gpickle files listed in a CSV.
    
    Args:
        csv_file_path: Path to CSV file with file names
        root_dir: Root directory for gpickle files
        
    Returns:
        Generator yielding tuples of (path, graph, pcode_map)
    """
    root_path = Path(root_dir)
    file_names, _ = read_csv(csv_file_path)
    
    for file_name in tqdm(file_names, desc="Processing Gpickle files"):
        prefix = file_name[:2]
        path = root_path / prefix / f"{file_name}.gpickle"
        
        if path.exists():
            try:
                with open(path, "rb") as fp:
                    G = pickle.load(fp)
                pcode_map = nx.get_node_attributes(G, "pcode")
                yield path, G, pcode_map
            except Exception as e:
                tqdm.write(f"[Error] Load Gpickle Failed {path}: {e}")
        else:
            tqdm.write(f"[Warning] File Not Found: {file_name}.gpickle")

# # benign and malwware
# def iterate_Gpickle(
#     csv_file_path: str | Path,
#     root_dir: str | Path
# ) -> Generator[Tuple[Path, nx.DiGraph, Dict[str, Sequence[str]]], None, None]:
#     """
#     Iterate through gpickle files listed in a CSV.

#     Args:
#         csv_file_path: Path to CSV file with file names
#         root_dir: Root directory for gpickle files

#     Returns:
#         Generator yielding tuples of (path, graph, pcode_map)
#     """
#     root_path = Path(root_dir)

#     # Read file names from CSV
#     csv_result = read_csv(csv_file_path)
#     file_names = csv_result[0]  # Get file names only

#     for file_name in tqdm(file_names, desc="Processing Gpickle files"):
#         # Try different directory structures
#         paths_to_try = [
#             # Try with benign/malware subdirectories with 2-char prefix
#             *[root_path / sub / file_name[:2] / f"{file_name}.gpickle" for sub in ("benign", "malware")],
#             # Try with benign/malware subdirectories without prefix
#             *[root_path / sub / f"{file_name}.gpickle" for sub in ("benign", "malware")],
#             # Try direct in root directory
#             root_path / f"{file_name}.gpickle",
#         ]

#         # Try each path
#         for path in paths_to_try:
#             if path.exists():
#                 try:
#                     with open(path, "rb") as fp:
#                         G = pickle.load(fp)
#                     pcode_map = nx.get_node_attributes(G, "pcode")
#                     yield path, G, pcode_map
#                     break  # Found and loaded successfully, move to next file
#                 except Exception as e:
#                     tqdm.write(f"[Error] Load Gpickle Failed {path}: {e}")
#         else:
#             # None of the paths worked
#             tqdm.write(f"[Warning] File Not Found: {file_name}.gpickle")

def read_csv(csv_file_path: str | Path) -> List[str]:
    """
    Read a CSV file and return file names and labels as a dictionary.
    
    Args:
        csv_file_path (str or Path): The path to the CSV file.
    
    Returns:
        Tuple[List[str], Dict[str, int]]: File names and a dictionary mapping file names to labels.
    """
    df = pd.read_csv(csv_file_path)
    file_names = df['file_name'].tolist()
    
    # Create a dictionary mapping file names to labels
    labels_dict = {}
    if 'label' in df.columns:
        for idx, row in df.iterrows():
            labels_dict[row['file_name']] = row['label']
    
    return file_names, labels_dict

def create_scheduler(optimizer, scheduler_type, **kwargs):
    """
    Create learning rate scheduler
    
    Args:
        optimizer: PyTorch optimizer
        scheduler_type: Type of scheduler ("step", "plateau", "cosine")
        **kwargs: Additional parameters for scheduler
        
    Returns:
        Learning rate scheduler
    """
    if scheduler_type == "step":
        step_size = kwargs.get("step_size", 30)
        gamma = kwargs.get("gamma", 0.5)
        return StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    elif scheduler_type == "plateau":
        patience = kwargs.get("patience", 10)
        factor = kwargs.get("factor", 0.5)
        min_lr = kwargs.get("min_lr", 1e-6)
        return ReduceLROnPlateau(optimizer, mode='min', patience=patience, 
                               factor=factor, min_lr=min_lr)
    
    elif scheduler_type == "cosine":
        T_max = kwargs.get("T_max", 100)
        eta_min = kwargs.get("eta_min", 1e-6)
        return CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    

def simple_early_stopping(val_acc, best_val_acc, patience_counter, patience):
    if val_acc > best_val_acc:
        return val_acc, 0, False  
    else:
        patience_counter += 1
        if patience_counter >= patience:
            return best_val_acc, patience_counter, True  
        return best_val_acc, patience_counter, False
    


def plot_training_curves(train_losses, val_accuracies, test_accuracies=None, val_losses=None):
    """繪製訓練曲線，包含 Validation 與 Test Accuracy"""
    plt.figure(figsize=(15, 5))
    
    # Training Loss
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Validation & Test Accuracy
    plt.subplot(1, 3, 2)
    plt.plot(val_accuracies, label='Val Acc')
    if test_accuracies:
        plt.plot(test_accuracies, label='Test Acc')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Validation Loss (optional)
    if val_losses:
        plt.subplot(1, 3, 3)
        plt.plot(val_losses, label='Val Loss')
        plt.title('Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()


def plot_confusion_matrix(cm, labels, filename="confusion_matrix.png"):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()


def test_model(model, test_loader, device, label_encoder):
    """測試模型並生成詳細結果"""
    model.eval()
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            pred = out.argmax(dim=1)
            y_true.extend(batch.y.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())
    
    # 計算指標
    accuracy = accuracy_score(y_true, y_pred)
    f1_micro = f1_score(y_true, y_pred, average='micro')
    f1_macro = f1_score(y_true, y_pred, average='macro')
    
    # 分類報告
    report = classification_report(y_true, y_pred, target_names=[str(c) for c in label_encoder.classes_])
    
    # 混淆矩陣
    original_labels = label_encoder.inverse_transform(sorted(set(y_true + y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=label_encoder.transform(original_labels))
    
    print("LabelEncoder classes:", label_encoder.classes_)
    print("y_true labels:", set(label_encoder.inverse_transform(y_true)))
    print("y_pred labels:", set(label_encoder.inverse_transform(y_pred)))
    print("y_pred counts:", dict(pd.Series(label_encoder.inverse_transform(y_pred)).value_counts()))


    return {
        'accuracy': accuracy,
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'classification_report': report,
        'confusion_matrix': cm,
        'original_labels': original_labels,
        'y_true': y_true,
        'y_pred': y_pred
    }


