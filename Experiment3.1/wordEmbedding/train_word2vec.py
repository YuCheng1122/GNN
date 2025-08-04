import pickle
from pathlib import Path
from gensim.models import Word2Vec
from tqdm import tqdm

OUTPUT_DIR = Path("/home/tommy/Projects/cross-architecture/Vector/20250509_new_train_450_token/model")

print("Loading sentences from pickle…")
with open("/home/tommy/Projects/cross-architecture/Vector/20250509_new_train_450_token/model/sentences_20250509_train_450.pkl", "rb") as f:
    raw_sentences = pickle.load(f)
print(f"Sample sentence: {raw_sentences[0]}")
sentences = list(tqdm(raw_sentences, desc="Preparing sentences"))

model = Word2Vec(
    sentences,
    vector_size=256,
    window=4,
    min_count=3,
    workers=48,
    seed=42,
)

print("Saving model…")
model.save(str(OUTPUT_DIR / "word2vec_20250509_train_450.model"))

# Check lens
print("Model lens: ",len(model.wv.key_to_index))

# model = Word2Vec(
#     vector_size=256,
#     window=4,
#     min_count=3,
#     workers=48,
#     seed=42,
# )

# model.build_vocab(
#     tqdm(sentences, desc="Building vocab"),
#     progress_per=10000
# )

# print("Training Word2Vec model…")
# for epoch in range(3):
#     print(f"Epoch {epoch+1}/3")
#     model.train(
#         corpus_iterable=tqdm(sentences, desc=f"Training epoch {epoch+1}"),
#         total_examples=model.corpus_count,
#         epochs=1,
#     )
# print("Saving model…")
# model.save(str(OUTPUT_DIR / "word2vec_20250509_train_450.model"))

# # Check lens
# print("Model lens: ",len(model.wv.key_to_index))

