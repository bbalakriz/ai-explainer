# face similarity search with pgvector and deepface

stores face embeddings (Facenet, 128 dimensions) in a pgvector enabled Postgres database and performs cosine similarity search to find matching faces.

## prerequisites

- podman (or docker) running locally
- python 3.12 (tensorflow does not support 3.14 yet)
- homebrew python: `/opt/homebrew/bin/python3.12`

## 1. start the pgvector container

```bash
podman run --name pgvector-container \
  -e POSTGRES_PASSWORD=mysecretpassword \
  -p 5432:5432 \
  -d pgvector/pgvector:pg16
```

verify it is running:

```bash
podman ps --filter name=pgvector-container
```

## 2. create the venv and install dependencies

```bash
/opt/homebrew/bin/python3.12 -m venv venv
source venv/bin/activate
pip install deepface pgvector psycopg2-binary Pillow requests tf-keras
```

## 3. set up the database table

```bash
podman exec pgvector-container psql -U postgres -c "
  CREATE EXTENSION IF NOT EXISTS vector;
  CREATE TABLE IF NOT EXISTS face_embeddings (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    image_path VARCHAR(500),
    embedding vector(128)
  );
  CREATE INDEX IF NOT EXISTS face_embeddings_embedding_idx
    ON face_embeddings USING hnsw (embedding vector_cosine_ops);
"
```

## 4. add images to the images folder

place face images in the `images/` directory. the script infers the person name from the filename:

- files containing `bean` are labelled as `bean`
- files containing `carrey` or `carry` are labelled as `carrey`
- anything else uses the filename stem as the label

```
images/
  bean1.jpg
  bean2.jpg
  carrey1.jpg
  carrey2.jpg
```

supported formats: png, jpg, jpeg, bmp, webp

## 5. insert images into pgvector

```bash
source venv/bin/activate
python download_and_insert.py
```

expected output:

```
step 1: scanning local images...
  bean: 2 images
  carrey: 2 images

step 2: connecting to pgvector database...

step 3: extracting embeddings and inserting into database...

  processing bean:
    extracting embedding from bean1.jpg...
    inserted bean1.jpg (128 dims)
    ...

done. inserted 4 face embeddings into pgvector.
```

## 6. search for similar faces

pass any image to the search script. it extracts the embedding and returns the closest matches from the database.

```bash
source venv/bin/activate
python search_faces.py images/bean1.jpg
```

optionally set the number of results (default 5):

```bash
python search_faces.py images/carrey1.jpg 3
```

expected output:

```
extracting embedding from: images/bean1.jpg
embedding extracted (128 dimensions)

searching for top 5 similar faces...

rank   name                 similarity   image
----------------------------------------------------------------------
1      bean                 1.000000     bean1.jpg
2      bean                 0.850000     bean2.jpg
3      carrey               0.620000     carrey1.jpg
4      carrey               0.600000     carrey2.jpg
```

## 7. verify database contents

```bash
podman exec pgvector-container psql -U postgres -c "SELECT id, name, image_path FROM face_embeddings;"
```

## notes

- the Facenet model weights are downloaded automatically on first run to `~/.deepface/weights/`
- `detector_backend="retinaface"` is used for accurate face detection and alignment on human photos
- cosine distance is used for the HNSW index which works well for normalized embeddings
- the default similarity threshold is 0.5; tune it based on your images
- to reset the database: `podman exec pgvector-container psql -U postgres -c "DELETE FROM face_embeddings;"`
