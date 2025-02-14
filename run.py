import configparser
import sqlite3
import hashlib
import subprocess
import platform
import logging
from pathlib import Path

from tqdm import tqdm
import numpy as np
import torch
from transformers import AutoProcessor, AutoTokenizer, CLIPModel
from PIL import Image
import faiss

def get_device() -> str:
    logger.info("Getting ML device")
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return "cpu"

def get_linux_image_viewer() -> str:
    logger.info("Getting image viewer")
    try:
        viewer = subprocess.check_output(['xdg-mime', 'query', 'default', 'image/png']).decode('utf-8').strip()

        if not viewer:
            logger.error("No default image viewer found.")
            return ''
        logger.info(f'viewer is {viewer}')
        return viewer
    except subprocess.CalledProcessError as e:
        logger.error(f"Error: {e}")
        return ''

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

def process_data_path(path: Path) -> bool:
    data_dir = to_absolute_path(path)
    if data_dir.exists() and not path.is_dir():
        logger.error(f"Error: {path} exists but is not a directory.")
        print(f"Error: {path} exists but is not a directory.")
        return False

    if not data_dir.exists():
        logger.info(f"{path} does not exist, so it will be creates")
        print(f"{path} does not exist, so it will be creates")
        data_dir.mkdir(parents=True, exist_ok=True)

def validate_image_path(path: Path) -> bool:
    if not path.exists():
        logger.error(f"Error: {path} does not exist")
        print(f"Error: {path} does not exist")
        return False
    if not path.is_dir():
        logger.error(f"Error: {path} is not a dir")
        print(f"Error: {path} is not a dir")
        return False
    return True


def open_file_list(file_list: list[Path] | list[str]) -> None:
    logger.info('opening files')
    logger.debug(f'file list is {file_list}')
    user_platform = platform.system()
    logger.info(f'user_platform is {user_platform}')

    if user_platform == "Darwin":
        subprocess.run(['open'] + file_list)
        return
    elif user_platform == "Windows":
        subprocess.run(['start'] + file_list)
        return
    elif user_platform == "Linux":
        image_viewer = get_linux_image_viewer()
        if image_viewer == '':
            print('Error: no image viewer found')
            return
        subprocess.run([image_viewer] + file_list)

def load_clip_model(clip_model) -> (CLIPModel, AutoProcessor, AutoTokenizer):
    logger.info('loading clip model')
    device = get_device()
    logger.info(f'device is {device}')
    model = CLIPModel.from_pretrained(clip_model).to(device)
    processor = AutoProcessor.from_pretrained(clip_model)
    tokenizer = AutoTokenizer.from_pretrained(clip_model)

    return model, processor, tokenizer

def embed_image(file: Path, model: CLIPModel, processor: AutoProcessor) -> np.array:
    logger.info(f'embedding image {file}')
    device = get_device()
    logger.info(f'ML device is {device}')
    with Image.open(file) as img:
        inputs = processor(images=[img], return_tensors="pt").to(device)
        image_features = model.get_image_features(**inputs).cpu().detach().numpy().flatten()
        logger.debug(f"image_features is {image_features}")
    return image_features

def embed_text(text: str, model: CLIPModel, tokenizer: AutoTokenizer) -> np.array:
    logger.info('embedding text {text}')
    device = get_device()
    logger.info(f'ML device is {device}')
    inputs = tokenizer([text], padding=True,return_tensors="pt").to(device)
    text_features = model.get_text_features(**inputs).cpu().detach().numpy().flatten()
    logger.debug(f'text_features is {text_features}')
    return text_features

def file_search(path: Path, database_file: Path) -> list:
    logger.info(f'running file search  on {path}')
    suffixes = {'.png', '.jpg', '.jpeg'} # I could make this more resilient with mimetypes or magic. I'm probably not going to, so don't use weird image formats
    image_files_on_disk = {str(f) for f in path.rglob("*") if f.suffix.lower() in suffixes}
    logger.debug(f'there are {len(image_files_on_disk)} image files in the database')
    logger.debug(f'all files on disk are: {image_files_on_disk}')
    image_files_in_database = set()

    with sqlite3.connect(database_file) as conn:
        logger.info('getting file list from database')
        cursor = conn.cursor()
        image_files_in_database = {t[0] for t in cursor.execute("SELECT path FROM Files;").fetchall()}
        logger.debug(f'there are {len(image_files_in_database)} files in the database')

    unindexed_files = image_files_on_disk - image_files_in_database
    logger.info(f'there are {len(unindexed_files)} files on disk')
    return unindexed_files

def index_path(path: Path, database_file: Path, index_file: Path, model: CLIPModel, processor: AutoProcessor) -> None:
    logger.info(f'Indexing {path}')
    new_files = set()
    new_vectors = None
    new_files = file_search(path=path, database_file=database_file)
    create_new_index = False
    logger.info('checking index file')
    if index_file.exists():
        logger.info('index_file exists, reading')
        index = faiss.read_index(str(index_file))
    else:
        logger.info(f"index file does not exist, a new index will be created at {index_file}")
        create_new_index = True

    count = 0
    for file in tqdm(new_files):
        logger.info(f'processing file {file}')
        vector = embed_image(file=file, model=model, processor=processor)
        vector_hash = hash_vector(vector)
        if create_new_index:
            logger.info('creating new index')
            d = vector.shape[0]
            logger.info(f'd is {d}')
            index = faiss.IndexFlatL2(d)
            create_new_index = False
        with sqlite3.connect(database_file) as conn:
            logger.info('inserting into database')
            cursor = conn.cursor()
            cursor.execute("INSERT INTO Files VALUES (?, ?)", (str(file), vector_hash, ))
            conn.commit()
        if new_vectors is None:
            logger.debug('creating new_vector matrix')
            new_vectors = np.zeros((len(new_files), vector.shape[0]))
        new_vectors[count] = vector
        count += 1

    if new_vectors is not None and len(new_vectors) > 0:
        logger.info('Adding new vectors to index')
        index.add(new_vectors)

    logger.info('Writing index to disk')
    faiss.write_index(index, str(index_file))
        

def search_loop(database_file: Path, index_file: Path, model: CLIPModel, processor: AutoProcessor, tokenizer: AutoTokenizer, search_depth: int, open_files: bool) -> None:
    logger.info("entering search loop")
    index = faiss.read_index(str(index_file))
    conn = sqlite3.connect(database_file)
    cursor = conn.cursor()
    while True:
        print('Type "exit" to exit')
        search_string = input("Type a string that you want to search. If the thing you type looks like that path to an image, it will be used for a similarity search.\n")
        logger.info(f"search string is {search_string}")
        if search_string.lower() == "exit":
            logger.info("Exiting from search loop")
            break
        elif Path(search_string).is_file():
            logger.info("search_string is a path, parsing as image to embed")
            search_vector = embed_image(file=Path(search_string), model=model, processor=processor)
        else:
            logger.info("search_string is a string, parsing as text to embed")
            search_vector = embed_text(text=search_string, model=model, tokenizer=tokenizer)
        search_vector = search_vector.reshape(1, -1)
        _, indexes = index.search(search_vector, search_depth)
        reconstructed_vectors = index.reconstruct_batch(indexes[0])
        hash_list = np.apply_along_axis(hash_vector, 1, reconstructed_vectors).tolist()
        results = list()
        logger.info('performing vector lookup')
        for vector_hash in hash_list:
            logger.debug(f'vector_hash is {vector_hash}')
            cursor.execute("SELECT path FROM Files WHERE vector_hash = ?", (vector_hash,))
            result = cursor.fetchall()
            logger.debug(f'result is {result}')
            results.append(result)
        for r in results:
            print(f"r in results is {r}")
        file_list = [f[0][0] for f in results]
        logger.debug(f'file_list is {file_list}')
        if open_files:
            open_file_list(file_list)

    conn.close()


def main():
    config_file = 'config.ini'
    config = configparser.ConfigParser()
    config.read(config_file)
    search_depth = int(config["ML"]["Search Depth"])
    open_files = config.getboolean("ML", "Open Files")
    logger.debug(f'search_depth is {search_depth}')
    logger.debug(f'open_files is {open_files}')

    index_paths = config["Storage"]["Path to Files"]
    if index_paths == '':
        logger.info('index_paths not configured')
        print('no index paths have been configured. Please enter the paths you would like to index. Enter "exit" to stop adding paths. If you do not enter any paths, the program will exit.')
        added_paths = list()
        while True:
            input_path = input()
            logger.info(f'input_path is {input_path}')
            if input_path.lower() == 'exit':
                if len(added_paths) == 0:
                    return
                break
            if not validate_image_path(Path(input_path)):
                print(f"Path does not appear to be valid, please try again")
                logger.info(f"Path does not appear to be valid")
            else:
                print(f'Adding {input_path}')
                added_paths.append(input_path)

        logger.info(f'input_paths is {added_paths}')
        config_value = ";".join(added_paths)
        logger.info(f'config value will be {config_value}')
        config.set("Storage", "Path to Files", config_value)
        with open(config_file, 'w') as f:
            config.write(f)
        index_paths = config_value

    index_paths = index_paths.split(";")
    index_paths = [to_absolute_path(Path(p)) for p in index_paths]
    logger.debug(f'index_paths is {index_paths}')

    clip_model = config["ML"]["CLIP Model"]
    logger.debug(f'clip_model is {clip_model}')
    model, processor, tokenizer = load_clip_model(clip_model=clip_model)
    model_hash = hash_string(clip_model)

    data_dir = config["Storage"]["Data Directory"]
    data_dir = data_dir / Path(model_hash)
    database_file = data_dir / Path('database.sqlite3')
    index_file = data_dir / Path('faiss.index')
    logger.debug(f'data_dir is {data_dir}')
    logger.debug(f'database_file is {database_file}')
    logger.debug(f'index_file is {index_file}')

    logger.debug('validating image directories')
    for path in index_paths:
        logger.debug(f'validating {path}')
        if not validate_image_path(path):
            logger.error(f"{path} is invalid. Please edit config.ini to correct this error")
            return

    logger.info('checking data directory')
    process_data_path(Path(data_dir))

    with sqlite3.connect(database_file) as conn:
        logger.info('creating Files tables')
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE IF NOT EXISTS Files (path text, vector_hash text);") 
        conn.commit()

    for path in index_paths:
        index_path(path, database_file=database_file, index_file=index_file, model=model, processor=processor)

    search_loop(database_file=database_file, index_file=index_file, model=model, processor=processor, tokenizer=tokenizer, search_depth=search_depth, open_files=open_files)

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    main()
