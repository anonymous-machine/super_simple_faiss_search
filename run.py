import configparser
import sqlite3
import hashlib
import subprocess
import platform
from pathlib import Path

from tqdm import tqdm
import numpy as np
import torch
from transformers import AutoProcessor, AutoTokenizer, CLIPModel
from PIL import Image
import faiss

def hash_string(string) -> str:
    hash_object = hashlib.sha256()
    hash_object.update(string.encode('utf-8'))
    return str(hash_object.hexdigest())

def hash_vector(vector) -> str:
    return hash_string(np.array2string(vector, separator=', ') )

def open_file_list(file_list: list[Path] | list[str]) -> None:
    try:
        match platform.system():
            case "Darwin":
                subprocess.run('open', file_list)
            case "Windows":
                subprocess.run(['start', '']  + file_list, shell=True)
            case "Linux":
                 subprocess.run(['xdg-open', file_list])
            case _:
                print("I don't know your OS, sorry.")
    except Exception as e:
        print(f"Error: {e}")

def load_clip_model(clip_model, device='cpu') -> (CLIPModel, AutoProcessor, AutoTokenizer):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = CLIPModel.from_pretrained(clip_model).to(device)
    processor = AutoProcessor.from_pretrained(clip_model)
    tokenizer = AutoTokenizer.from_pretrained(clip_model)

    return model, processor, tokenizer

def embed_image(file: Path, model: CLIPModel, processor: AutoProcessor) -> np.array:
    with Image.open(file) as img:
        inputs = processor(images=[img], return_tensors="pt")
        image_features = model.get_image_features(**inputs).detach().numpy().flatten()
    return image_features

def embed_text(text: str, model: CLIPModel, tokenizer: AutoTokenizer) -> np.array:
    inputs = tokenizer([text], padding=True,return_tensors="pt")
    text_features = model.get_text_features(**inputs).detach().numpy().flatten()
    return text_features

def file_search(root, cursor):
    suffixes = {'.png', '.jpg', '.jpeg'} # I could make thie more resilient with mimetypes or magic. I'm probably not going to, so don't use weird image formats
    image_files_on_disk = {f for f in root.rglob("*") if f.suffix.lower() in suffixes}
    image_files_in_database = {t[0] for t in cursor.execute("SELECT path FROM Files;").fetchall()}
    unindexed_files = image_files_on_disk - image_files_in_database
    return unindexed_files

def main():
    config = configparser.ConfigParser()
    root = config["Storage"]["Path to Files"]
    data_dir = config["Storage"]["Data Directory"]
    clip_model = config["ML"]["CLIP Model"]

    print('Checking path to images...')
    root = Path(root)
    if root.exists() and not root.is_dir():
        print('Error: Path to Images should be a folder.')
        return
    if not root.exists():
        print('Error: Path to Images does not appear to be correct.')
        return

    print('Checking Data Directory...')
    data_dir = Path(data_dir)
    if data_dir.exists() and not data_dir.is_dir():
        print('Error: Data Directory should be a folder')
        return
    if not data_dir.exists():
        print('Data Directory does not exist, so it will be created.')
        data_dir.mkdir(parents=True, exist_ok=True)

    model_hash = hash_string(clip_model)
    data_dir = data_dir / Path(model_hash)
    if not data_dir.exists():
        data_dir.mkdir(parents=True,exist_okay=True)
    sqlite_file = data_dir / Path('database.sqlite3')
    index_file = data_dir / Path('faiss.index')
    root = Path.cwd() #this should really be user configurable

    # misc loading
    model, processor, tokenizer = load_clip_model(model=clip_model)
    index = None
    if index_file.exists():
        index = faiss.read_index(index_file)

    conn = sqlite3.connect(sqlite_file)
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS Files (path text, vector_hash text);") 
    conn.commit()

    new_files = set()
    print('Bringing in new files...')
    new_files = file_search(root=root, cursor=cursor)
    print(f'{len(new_files)} files found')
    print(new_files)

    new_vectors = None
    count = 0
    for file in tqdm(new_files):
        vector = embed_image(file=file, model=model, processor=processor)
        #print(vector)
        vector_hash = hash_vector(vector)
        #print(vector_hash)
        if index is None:
            tqdm.write("Index file does not exist. A new index will be created")
            print(vector.shape)
            #return 0
            d = vector.shape[0] 
            index = faiss.IndexFlatL2(d)
        #index.add(vector)
        cursor.execute("INSERT INTO Files VALUES (?, ?)", (str(file), vector_hash,))
        if new_vectors is None:
            new_vectors = np.zeros((len(new_files), vector.shape[0]))
        new_vectors[count] = vector
        #conn.commit()
        count += 1
    if len(new_vectors) > 0:
        print(new_vectors.shape)
        #print(new_vectors)
        #arr = np.concatenate(new_vectors, axis=0)
        #print(arr.shape)
        index.add(new_vectors)

    while True:
        print('Type "N" to exit')
        search_string = input("Type a string that you want to search. If the thing you type looks like that path to an image, it will be used for a similarity search.\n")
        if search_string.upper() == "N":
            break
        elif Path(search_string).is_file():
            search_vector = embed_image(file=Path(search_string), model=model, processor=processor)
        else:
            search_vector = embed_text(text=search_string, model=model, tokenizer=tokenizer)
        search_vector = search_vector.reshape(1, -1)
        _, indexes = index.search(search_vector, 10)
        reconstructed_vectors = index.reconstruct_batch(indexes[0])
        hash_list = np.apply_along_axis(hash_vector, 1, reconstructed_vectors).tolist()
        results = list()
        for vector_hash in hash_list:
            cursor.execute("SELECT path FROM Files WHERE vector_hash = ?", (vector_hash,))
            result = cursor.fetchall()
            results.append(result)
        for r in results:
            print(r)
        print(results)
        # I think you use faiss.reconstruct_batch(I) to get the original vectors back
        # then hash the vectors
        # then do a lookup of the corresponding filepaths from the sqlite database
        # and then open those images


if __name__ == '__main__':
    main()
