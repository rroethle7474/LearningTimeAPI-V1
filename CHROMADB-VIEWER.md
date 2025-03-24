# ChromaDB Viewer Guide for Windows

This guide provides instructions on how to view and explore the contents of a local ChromaDB database on Windows.

## Methods for Viewing ChromaDB Contents

### 1. Chroma UI (Official Web Interface)

Chroma UI is the official web-based interface for ChromaDB:

1. Install via pip:
   ```bash
   pip install chromadb-admin
   ```

2. Run the admin interface by pointing to your ChromaDB directory:
   ```bash
   chromadb-admin --path ./chromadb
   ```

3. This will start a local web server where you can browse your collections through a browser interface.

### 2. ChromaDB Explorer (Third-party GUI Tool)

ChromaDB Explorer is a dedicated desktop application for Windows:

1. Download from GitHub: [https://github.com/blip-solutions/chromadb-explorer](https://github.com/blip-solutions/chromadb-explorer)
2. Install the application
3. Launch and connect to your local ChromaDB instance
4. Browse collections, view embeddings, and explore metadata

This option provides a user experience similar to DB Browser for SQLite or pgAdmin for PostgreSQL.

### 3. Programmatic Approach (Python Script)

You can create a simple Python script to inspect your ChromaDB:

```python
import chromadb

# Connect to your existing ChromaDB
client = chromadb.PersistentClient(path="./chromadb")

# List all collections
collections = client.list_collections()
print(f"Collections: {[c.name for c in collections]}")

# For each collection, you can get items
for collection in collections:
    coll = client.get_collection(collection.name)
    # Get all items (be careful if you have a large collection)
    items = coll.get()
    print(f"Collection '{collection.name}' has {len(items['ids'])} items")
    # Print a sample of items
    if len(items['ids']) > 0:
        print(f"Sample item ID: {items['ids'][0]}")
        print(f"Sample metadata: {items['metadatas'][0] if items['metadatas'] else 'No metadata'}")
```

Save this as `view_chromadb.py` and run it with:

```bash
python view_chromadb.py
```

## Recommendation

For Windows users, ChromaDB Explorer is recommended as it provides the most user-friendly interface similar to other database management tools you may be familiar with.

## Notes

- The ChromaDB path in this project is configured in the `.env` file via the `CHROMADB_PATH` setting
- The default location is `./chromadb` in the project root directory
- Make sure you have the correct path when using any of these viewing methods
