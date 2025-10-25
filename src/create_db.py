import os
import chromadb
from tqdm import tqdm
from datasets import load_dataset
from huggingface_hub import login
from sentence_transformers import SentenceTransformer
from src import DB, MODEL_NAME

hf_token = os.getenv("HF_TOKEN")

login(hf_token, add_to_git_credential=True)
HF_USER = "NEGU93"
DATASET_NAME = f"{HF_USER}/pricer-data"


def description(item):
    text = item["text"].replace(
        "How much does this cost to the nearest dollar?\n\n", ""
    )
    return text.split("\n\nPrice is $")[0]


def delete_collection_if_exists(client, collection_name):
    # Check if the collection exists and delete it if it does
    existing_collections = client.list_collections()
    existing_names = [c.name for c in existing_collections]

    if collection_name in existing_names:
        client.delete_collection(collection_name)
        print(f"Deleted existing collection: {collection_name}")


if __name__ == "__main__":
    client = chromadb.PersistentClient(path=DB)
    collection_name = "products"
    delete_collection_if_exists(client, collection_name)

    collection = client.create_collection(collection_name)

    # Get the model for vector encoding
    model = SentenceTransformer(MODEL_NAME)

    # Get dataset
    dataset = load_dataset(DATASET_NAME)
    train = dataset["train"]

    batch_size = 1000
    for i in tqdm(range(0, len(train), batch_size), desc="Processing batches"):
        batch = train.select(range(i, min(i + batch_size, len(train))))
        documents = [description(item) for item in batch]
        vectors = (
            model.encode(documents, show_progress_bar=False)
            .astype(float)
            .tolist()
        )
        metadatas = [{"price": item["price"]} for item in batch]
        ids = [f"doc_{j}" for j in range(i, i + len(documents))]
        collection.add(
            ids=ids,
            documents=documents,
            embeddings=vectors,
            metadatas=metadatas,
        )
