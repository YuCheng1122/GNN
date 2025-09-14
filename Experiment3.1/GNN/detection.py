import torch
import random
import numpy as np
from torch_geometric.loader import DataLoader
from train_utils import load_or_cache_data, train_epoch, evaluate
from utils import simple_early_stopping, create_scheduler, test_model
from model import GCNBinary

def set_random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def run_experiment(seed):
    set_random_seed(seed)
    
    train_csv_path = "/home/tommy/Projects/pcodeFcg/dataset/csv/temp/train_detection.csv"
    test_csv_path = "/home/tommy/Projects/pcodeFcg/dataset/csv/temp/test_detection.csv"
    train_dir = "/home/tommy/Projects/pcodeFcg/vector/contrastive/GNN/arm_cbow_v2/train"
    test_dir = "/home/tommy/Projects/pcodeFcg/vector/contrastive/GNN/arm_cbow_v2/test"
    batch_size = 32
    hidden_channels = 64
    lr = 0.01
    epochs = 200
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_graphs, val_graphs, test_graphs, label_encoder, num_classes = load_or_cache_data(
        train_csv_path, test_csv_path, train_dir, test_dir,
        cache_file="/home/tommy/Projects/pcodeFcg/vector/contrastive/GNN/arm_cbow_v2/processed_data_cache.pkl",
        force_reload=False
    )

    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)

    model = GCNBinary(num_node_features=256, hidden_channels=hidden_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()
    scheduler = create_scheduler(optimizer, "plateau")

    best_val_acc = 0
    patience = 15
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        train_epoch(model, train_loader, optimizer, criterion, device)
        val_accuracy, val_loss = evaluate(model, val_loader, device)
        test_accuracy, _ = evaluate(model, test_loader, device)

        scheduler.step(val_loss)

        best_val_acc, patience_counter, should_stop = simple_early_stopping(
            val_accuracy, best_val_acc, patience_counter, patience
        )

        if should_stop:
            break

    # 測試模型
    test_results = test_model(model, test_loader, device, label_encoder)
    return test_results


def main():
    seeds = [42, 123, 2025, 31415, 8888]
    all_results = []

    for i, seed in enumerate(seeds):
        print(f"\n 第 {i+1} 次實驗，Seed = {seed}")
        results = run_experiment(seed)
        all_results.append(results)

        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"F1-micro: {results['f1_micro']:.4f}")
        print(f"F1-macro: {results['f1_macro']:.4f}")

    # 計算平均與標準差
    avg_acc = np.mean([r['accuracy'] for r in all_results])
    avg_f1_micro = np.mean([r['f1_micro'] for r in all_results])
    avg_f1_macro = np.mean([r['f1_macro'] for r in all_results])

    std_acc = np.std([r['accuracy'] for r in all_results])
    std_f1_micro = np.std([r['f1_micro'] for r in all_results])
    std_f1_macro = np.std([r['f1_macro'] for r in all_results])

    print("\n📊 五次實驗平均結果：")
    print(f"Accuracy     : {avg_acc:.4f} ± {std_acc:.4f}")
    print(f"F1-score (micro): {avg_f1_micro:.4f} ± {std_f1_micro:.4f}")
    print(f"F1-score (macro): {avg_f1_macro:.4f} ± {std_f1_macro:.4f}")


if __name__ == "__main__":
    main()
