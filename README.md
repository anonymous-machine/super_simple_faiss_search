# Limitations

This program does not have the ability to track moved files. If you move or
rename a file, it will be indexed twice.

There is not currently a way to remove an indexed path other than deleting the
the ``sqlite.database`` and ``faiss.index`` files, and recreating the index.
This is a limitation of the FAISS library.

If you want to add more paths to index after the first run, edit the config.ini file. Separate paths using ``;``

# Using

1. Add the path you wish to index to config.ini

2. Run run.py

# Installation

## Method 1: Using ``uv``

I recommend using uv to create a virtual environment just for this program. Instructions are here.

1. Install [uv](https://docs.astral.sh/uv/#installation)

2. Navigate to the directory containing run.py

3. Create the virtual env

```bash
uv venv create .
```

4. Install requirements

```bash
uv pip install -r requirements.txt
```

5. Run the program

```bash
uv run run.py 
```

## Method 2: Plain Python (no venv)

1. Navigate to the firectory containing run.py

2. Install requirements

```bash
pip install -r requirements.txt
```

or

```bash
pip3 install -r requirements.txt
```

3. Run the program

```bash
python run.py 
```

or

```bash
python3 run.py
```
