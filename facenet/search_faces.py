"""
searches for similar faces in the pgvector database given an input image.
extracts a facenet embedding from the query image and performs a cosine
similarity search against stored face embeddings.
"""

import sys
import os
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


def get_embedding(image_path):
    """extract a 128 dim facenet embedding from an image."""
    result = DeepFace.represent(
        img_path=image_path,
        model_name="Facenet",
        enforce_detection=False,
        detector_backend="retinaface",
    )
    return result[0]["embedding"]


def search_similar_faces(conn, query_embedding, top_k=5, threshold=0.5):
    """find the top k faces above the similarity threshold using cosine distance."""
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, name, image_path,
               1 - (embedding <=> %s::vector) AS similarity
        FROM face_embeddings
        WHERE 1 - (embedding <=> %s::vector) >= %s
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """,
        (query_embedding, query_embedding, threshold, query_embedding, top_k),
    )
    results = cur.fetchall()
    cur.close()
    return results


def main():
    if len(sys.argv) < 2:
        print("usage: python search_faces.py <image_path> [top_k] [threshold]")
        print("  image_path  path to the query image")
        print("  top_k       number of results to return (default: 5)")
        print("  threshold   minimum similarity to include (default: 0.78)")
        sys.exit(1)

    image_path = sys.argv[1]
    top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5

    if not os.path.exists(image_path):
        print(f"error: file not found: {image_path}")
        sys.exit(1)

    print(f"extracting embedding from: {image_path}")
    query_embedding = get_embedding(image_path)
    print(f"embedding extracted ({len(query_embedding)} dimensions)")

    print(f"\nsearching for top {top_k} similar faces (threshold >= {threshold})...")
    conn = psycopg2.connect(**DB_CONFIG)
    register_vector(conn)

    results = search_similar_faces(conn, query_embedding, top_k, threshold)
    conn.close()

    if not results:
        print("no matching faces found above the similarity threshold.")
        return

    print(f"\n{'rank':<6} {'name':<20} {'similarity':<12} {'image'}")
    print("-" * 70)
    for i, (face_id, name, img_path, similarity) in enumerate(results, 1):
        basename = os.path.basename(img_path) if img_path else "n/a"
        print(f"{i:<6} {name:<20} {similarity:<12.6f} {basename}")


if __name__ == "__main__":
    main()
