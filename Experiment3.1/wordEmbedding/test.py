from gensim.models import Word2Vec
import pickle
from tqdm import tqdm

# 1. 載入訓練好的 model
model_path = "/home/tommy/Projects/cross-architecture/Vector/20250509_new_train_450/model/word2vec_20250509_train_450.model"
model = Word2Vec.load(model_path)

# 2. 載入測試語料（list of list of tokens）
test_pkl = "/home/tommy/Projects/cross-architecture/Vector/20250509_new_test_600/model/sentences_20250509_test_600.pkl"
with open(test_pkl, "rb") as f:
    test_sentences = pickle.load(f)

# 3. 計算 coverage
total_tokens = 0
covered_tokens = 0
for sent in tqdm(test_sentences, desc="Processing sentences"):
    for token in sent:
        total_tokens += 1
        if token in model.wv:
            covered_tokens += 1

# 4. 計算 type-level
unique_test_tokens = set(token for sent in test_sentences for token in sent)
in_vocab_types = {token for token in unique_test_tokens if token in model.wv}

# 5. 輸出結果
print(f"Token-level coverage: {covered_tokens}/{total_tokens} = {covered_tokens/total_tokens*100:.2f}%")
print(f"Type-level coverage: {len(in_vocab_types)}/{len(unique_test_tokens)} = {len(in_vocab_types)/len(unique_test_tokens)*100:.2f}%")

# # 6. 顯示沒有包含到的 tokens（type-level）
# out_of_vocab_types = unique_test_tokens - in_vocab_types
# print(f"\nOut-of-vocabulary types ({len(out_of_vocab_types)}):")
# for token in sorted(out_of_vocab_types):
#     print(token)
