from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from sentence_transformers import SentenceTransformer

DATA_PATH = Path(__file__).resolve().parent / "cleve_data.pkl"

app = FastAPI()
model = SentenceTransformer("all-MiniLM-L6-v2")
df = pd.read_pickle(DATA_PATH)
emb_matrix = np.vstack(df["embedding"].to_list())
emb_norms = np.linalg.norm(emb_matrix, axis=1)


@app.get("/")
def index():
    return {"ok": True}



def find_best_painting(
    mood,
    atmosphere,
    setting,
    style,
    df,
    model,
    text_col="description",
    image_col="image_web",
    embedding_col="embedding",
):
    if df.empty:
        raise ValueError("df is empty")

    user_text = " ".join([mood, atmosphere, setting, style]).strip()
    user_embedding = np.array(model.encode(user_text))
    user_norm = np.linalg.norm(user_embedding)

    if user_norm == 0:
        raise ValueError("User embedding has zero norm")

    denom = user_norm * emb_norms

    similarities = np.full(len(df), -1.0, dtype=float)
    valid = denom != 0
    similarities[valid] = (emb_matrix[valid] @ user_embedding) / denom[valid]

    best_index = int(np.argmax(similarities))
    best_row = df.iloc[best_index]

    return {
        "id": int(best_row["id"]) if "id" in best_row and pd.notna(best_row["id"]) else None,
        "description": best_row.get(text_col),
        "image_url": best_row.get(image_col),
        "similarity": float(similarities[best_index]),
        "query": user_text,
    }


@app.get("/recommend")
def recommend(
    mood: str = "",
    atmosphere: str = "",
    setting: str = "",
    style: str = "",
):
    query_text = " ".join([mood, atmosphere, setting, style]).strip()
    if not query_text:
        raise HTTPException(status_code=400, detail="Provide at least one field")
    return find_best_painting(
        mood=mood,
        atmosphere=atmosphere,
        setting=setting,
        style=style,
        df=df,
        model=model,
        embedding_col="embedding",
    )
