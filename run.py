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

def get_device() -> str:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return "cpu"

def hash_string(string) -> str:
    hash_object = hashlib.sha256()
    hash_object.update(string.encode('utf-8'))
    return str(hash_object.hexdigest())

def hash_vector(vector) -> str:
    return hash_string(np.array2string(vector, separator=', ') )

def to_absolute_path(path: Path) -> Path:
    if not path.is_absolute():
        return path.resolve()
    return path

def open_file_list(file_list: list[Path] | list[str]) -> None:
    print('opening files')
    #print(file_list)
    try:
        match platform.system():
            case "Darwin":
                prefix = ['open']
            case "Windows":
                prefix = ['start']
            case "Linux":
                prefix = ['xdg-open']
            case _:
                print("I don't know your OS, sorry.")
    except Exception as e:
        print(f"Error opening files: {e}")

    subprocess.run(prefix + file_list)

def load_clip_model(clip_model) -> (CLIPModel, AutoProcessor, AutoTokenizer):
    device = get_device()
    model = CLIPModel.from_pretrained(clip_model).to(device)
    processor = AutoProcessor.from_pretrained(clip_model)
    tokenizer = AutoTokenizer.from_pretrained(clip_model)

    return model, processor, tokenizer

def embed_image(file: Path, model: CLIPModel, processor: AutoProcessor) -> np.array:
    device = get_device()
    with Image.open(file) as img:
        inputs = processor(images=[img], return_tensors="pt").to(device)
        image_features = model.get_image_features(**inputs).cpu().detach().numpy().flatten()
    return image_features

def embed_text(text: str, model: CLIPModel, tokenizer: AutoTokenizer) -> np.array:
    device = get_device()
    inputs = tokenizer([text], padding=True,return_tensors="pt").to(device)
    text_features = model.get_text_features(**inputs).cpu().detach().numpy().flatten()
    return text_features

def file_search(root, cursor):
    suffixes = {'.png', '.jpg', '.jpeg'} # I could make thie more resilient with mimetypes or magic. I'm probably not going to, so don't use weird image formats
    image_files_on_disk = {str(f) for f in root.rglob("*") if f.suffix.lower() in suffixes}
    image_files_in_database = {t[0] for t in cursor.execute("SELECT path FROM Files;").fetchall()}
    unindexed_files = image_files_on_disk - image_files_in_database
    return unindexed_files

def main():
    config = configparser.ConfigParser()
    config.read('config.ini')
    root = config["Storage"]["Path to Files"]
    data_dir = config["Storage"]["Data Directory"]
    clip_model = config["ML"]["CLIP Model"]
    search_depth = int(config["ML"]["Search Depth"])

    print('Checking path to images...')
    root = to_absolute_path(Path(root))
    if root.exists() and not root.is_dir():
        print('Error: Path to Images should be a folder.')
        return
    if not root.exists():
        print('Error: Path to Images does not appear to be correct.')
        return

    print('Checking Data Directory...')
    data_dir = to_absolute_path(Path(data_dir))
    if data_dir.exists() and not data_dir.is_dir():
        print('Error: Data Directory should be a folder')
        return
    if not data_dir.exists():
        print('Data Directory does not exist, so it will be created.')
        data_dir.mkdir(parents=True, exist_ok=True)

    model_hash = hash_string(clip_model)
    data_dir = data_dir / Path(model_hash)
    if not data_dir.exists():
        data_dir.mkdir(parents=True,exist_ok=True)
    sqlite_file = data_dir / Path('database.sqlite3')
    index_file = data_dir / Path('faiss.index')
    root = Path.cwd() #this should really be user configurable

    # misc loading
    model, processor, tokenizer = load_clip_model(clip_model=clip_model)
    index = None
    if index_file.exists():
        index = faiss.read_index(str(index_file))

    conn = sqlite3.connect(sqlite_file)
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS Files (path text, vector_hash text);") 
    conn.commit()

    new_files = set()
    print('Bringing in new files...')
    new_files = file_search(root=root, cursor=cursor)

    new_vectors = None
    count = 0
    for file in tqdm(new_files):
        vector = embed_image(file=file, model=model, processor=processor)
        vector_hash = hash_vector(vector)
        if index is None:
            tqdm.write("Index file does not exist. A new index will be created")
            d = vector.shape[0] 
            index = faiss.IndexFlatL2(d)
        cursor.execute("INSERT INTO Files VALUES (?, ?)", (str(file), vector_hash,))
        conn.commit()
        if new_vectors is None:
            new_vectors = np.zeros((len(new_files), vector.shape[0]))
        new_vectors[count] = vector
        count += 1
    if new_vectors is not None and len(new_vectors) > 0:
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
        _, indexes = index.search(search_vector, search_depth)
        reconstructed_vectors = index.reconstruct_batch(indexes[0])
        hash_list = np.apply_along_axis(hash_vector, 1, reconstructed_vectors).tolist()
        results = list()
        for vector_hash in hash_list:
            cursor.execute("SELECT path FROM Files WHERE vector_hash = ?", (vector_hash,))
            result = cursor.fetchall()
            results.append(result)
        for r in results:
            print(f"r in results is {r}")
        file_list = [f[0][0] for f in results]
        #print(file_list)
        open_file_list(file_list)

    faiss.write_index(index, str(index_file))
    conn.commit()
    conn.close()

if __name__ == '__main__':
    main()
