import networkx as nx
import torch
from torch_geometric.loader import DataLoader
import sklearn.preprocessing as labelEncoder
from torch_geometric.utils import from_networkx
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from utils import read_csv, iterate_Gpickle

def load_data(csv_file_path, root_dir, vector_dim=256):
    """
    Load graph data from gpickle files and prepare for dataloader.
    
    Args:
        csv_file_path: Path to CSV file with file names and labels
        root_dir: Root directory for gpickle files
        transform: Optional transform to apply to graphs
        
    Returns:
        Tuple[List[Data], List[int]]: List of PyG Data objects and corresponding labels
    """

    graph_list = []
    labels = []
    
    # Get file names and labels from CSV
    file_names, labels_dict = read_csv(csv_file_path)
    
    # Iterate through gpickle files
    for path, G, pcode_map in iterate_Gpickle(csv_file_path, root_dir):
        try:
            file_name = path.stem
            label = str(labels_dict.get(file_name, "unknown"))
    
            for node in G.nodes():
                vec = G.nodes[node].get("vector")
                if not isinstance(vec, np.ndarray) or vec.size != vector_dim:
                    vec = np.zeros(vector_dim, dtype=np.float32)
                # G.nodes[node]["x"] = torch.tensor(vec, dtype=torch.float32)
            data = from_networkx(G,group_node_attrs=["vector"])
            data.x = data.x.float()
            del data.vector
            graph_list.append(data)
            labels.append(label)
    
        except Exception as e:
            print(f"[ERROR] {path}: {e}")

    return graph_list, labels

# def load_data(csv_file_path, root_dir, vector_dim=256):
#     """
#     Load graph data from gpickle files and prepare for dataloader.
    
#     Args:
#         csv_file_path: Path to CSV file with file names and labels
#         root_dir: Root directory for gpickle files
#         transform: Optional transform to apply to graphs
        
#     Returns:
#         Tuple[List[Data], List[int]]: List of PyG Data objects and corresponding labels
#     """
#     graph_list = []
#     labels = []
    
#     # Get file names and labels from CSV
#     file_names, labels_dict = read_csv(csv_file_path)
#     for path, G, pcode_map in iterate_Gpickle(csv_file_path, root_dir):
#         try:
#             file_name = path.stem
#             label = str(labels_dict.get(file_name, "unknown"))

#             # 建立 node features list
#             vectors = []
#             for node in G.nodes():
#                 vec = G.nodes[node].get("vector")
#                 if not isinstance(vec, np.ndarray) or vec.size != vector_dim:
#                     vec = np.zeros(vector_dim, dtype=np.float32)
#                 vectors.append(vec)

#             vectors = np.array(vectors, dtype=np.float32) 

#             data = from_networkx(G)
#             data.x = torch.tensor(vectors, dtype=torch.float32)  
#             graph_list.append(data)
#             labels.append(label)

#         except Exception as e:
#             print(f"[ERROR] {path}: {e}")
#     return graph_list, labels
    

def load_or_cache_data(train_csv_path, test_csv_path, train_dir, test_dir, 
                       cache_file, val_size=0.2, random_state=42, force_reload=False):
    """
    Load or cache processed train/validation/test data
    
    Args:
        train_csv_path: Path to training CSV file
        test_csv_path: Path to test CSV file
        train_dir: Training data directory
        test_dir: Test data directory
        cache_file: Cache file name
        val_size: Validation set ratio
        random_state: Random seed
        force_reload: Force reload data from scratch
    
    Returns:
        tuple: (train_graphs, val_graphs, test_graphs, label_encoder, num_classes)
    """
    
    if force_reload and os.path.exists(cache_file):
        os.remove(cache_file)
    
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            cached_data = pickle.load(f)
            return (cached_data['train_graphs'], 
                   cached_data['val_graphs'],
                   cached_data['test_graphs'], 
                   cached_data['label_encoder'], 
                   cached_data['num_classes'])
    
    # Load raw data
    train_graphs, train_labels = load_data(train_csv_path, train_dir)
    test_graphs, test_labels = load_data(test_csv_path, test_dir)
    
    # Split train/validation
    train_graphs, val_graphs, train_labels, val_labels = train_test_split(
        train_graphs, train_labels, test_size=val_size, 
        stratify=train_labels, random_state=random_state
    )
    
    # test_graphs, val_graphs, test_labels, val_labels = train_test_split(
    #     test_graphs, test_labels, test_size=val_size, 
    #     stratify=test_labels, random_state=random_state
    # )
    
    # Label encoding
    label_encoder = LabelEncoder()
    label_encoder.fit(train_labels + val_labels + test_labels)
    
    encoded_train_labels = label_encoder.transform(train_labels)
    encoded_val_labels = label_encoder.transform(val_labels)
    encoded_test_labels = label_encoder.transform(test_labels)
    
    num_classes = len(label_encoder.classes_)
    
    # Update graph labels
    for i, data in enumerate(train_graphs):
        data.y = torch.tensor(encoded_train_labels[i], dtype=torch.long)
        # print("Classes:", label_encoder.classes_)
        
    for i, data in enumerate(val_graphs):
        data.y = torch.tensor(encoded_val_labels[i], dtype=torch.long)
        # print("Classes:", label_encoder.classes_)
    for i, data in enumerate(test_graphs):
        data.y = torch.tensor(encoded_test_labels[i], dtype=torch.long)
        # print("Classes:", label_encoder.classes_)

    # Save processed data
    cache_data = {
        'train_graphs': train_graphs,
        'val_graphs': val_graphs,
        'test_graphs': test_graphs,
        'label_encoder': label_encoder,
        'num_classes': num_classes
    }
    
    with open(cache_file, 'wb') as f:
        pickle.dump(cache_data, f)
    
    return train_graphs, val_graphs, test_graphs, label_encoder, num_classes


def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    
    return total_loss / len(train_loader.dataset)


def evaluate(model, data_loader, device):
    model.eval()
    correct = 0
    total_loss = 0
    criterion = torch.nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, batch.y)
            pred = out.argmax(dim=1)
            correct += int((pred == batch.y).sum())
            total_loss += loss.item() * batch.num_graphs
    
    accuracy = correct / len(data_loader.dataset)
    avg_loss = total_loss / len(data_loader.dataset)
    return accuracy, avg_loss