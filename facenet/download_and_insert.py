"""
scans the images/ folder for face images, extracts face embeddings
using deepface/facenet and inserts them into a pgvector enabled postgres database.

place images in the images/ folder with filenames that contain the person name:
  - files containing 'bean' are labelled as bean
  - files containing 'carrey' or 'carry' are labelled as carrey
  - anything else gets the filename stem as its label
"""

import os
import sys
import psycopg2
import numpy as np
from pgvector.psycopg2 import register_vector
from deepface import DeepFace

DB_CONFIG = {
    "host": "localhost",
    "dbname": "postgres",
    "user": "postgres",
    "password": "mysecretpassword",
}

IMAGES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".gif"}


def guess_character(filename):
    """infer person name from the filename."""
    name = filename.lower()
    if "bean" in name:
        return "bean"
    if "carrey" in name or "carry" in name:
        return "carrey"
    return os.path.splitext(filename)[0]


def scan_local_images():
    """find all supported images in the images directory grouped by character."""
    if not os.path.isdir(IMAGES_DIR):
        print(f"images directory not found: {IMAGES_DIR}")
        print("create it and add some character images first.")
        sys.exit(1)

    grouped = {}
    for fname in sorted(os.listdir(IMAGES_DIR)):
        ext = os.path.splitext(fname)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            continue
        character = guess_character(fname)
        grouped.setdefault(character, []).append(os.path.join(IMAGES_DIR, fname))

    return grouped


def get_embedding(image_path):
    """extract a 128 dim facenet embedding from an image."""
    try:
        result = DeepFace.represent(
            img_path=image_path,
            model_name="Facenet",
            enforce_detection=False,
            detector_backend="retinaface",
        )
        return result[0]["embedding"]
    except Exception as e:
        print(f"  error extracting embedding from {image_path}: {e}")
        return None


def setup_database(conn):
    """ensure the pgvector extension and face_embeddings table exist."""
    cur = conn.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS face_embeddings (
            id SERIAL PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            image_path VARCHAR(500),
            embedding vector(128)
        )
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS face_embeddings_embedding_idx
        ON face_embeddings USING hnsw (embedding vector_cosine_ops)
    """)
    conn.commit()
    cur.close()


def insert_face(conn, name, image_path, embedding):
    """insert a single face embedding into the database."""
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO face_embeddings (name, image_path, embedding) VALUES (%s, %s, %s)",
        (name, image_path, embedding),
    )
    conn.commit()
    cur.close()


def main():
    print("step 1: scanning local images...")
    grouped = scan_local_images()

    if not grouped:
        print("no supported images found in images/ folder.")
        sys.exit(1)

    for character, paths in grouped.items():
        print(f"  {character}: {len(paths)} images")

    print("\nstep 2: connecting to pgvector database...")
    conn = psycopg2.connect(**DB_CONFIG)
    register_vector(conn)
    setup_database(conn)

    # clear existing entries for a clean run
    cur = conn.cursor()
    cur.execute("DELETE FROM face_embeddings")
    conn.commit()
    cur.close()

    print("\nstep 3: extracting embeddings and inserting into database...")
    total_inserted = 0

    for character, paths in grouped.items():
        print(f"\n  processing {character}:")
        for path in paths:
            print(f"    extracting embedding from {os.path.basename(path)}...")
            embedding = get_embedding(path)
            if embedding is None:
                continue

            insert_face(conn, character, path, embedding)
            total_inserted += 1
            print(f"    inserted {os.path.basename(path)} ({len(embedding)} dims)")

    conn.close()
    print(f"\ndone. inserted {total_inserted} face embeddings into pgvector.")


if __name__ == "__main__":
    main()
