from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import os
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http import models
import uuid

DATA_PATH = "/opt/airflow/data"
COLLECTION_NAME = "documents"
QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
EMBEDDING_MODEL = "cointegrated/rubert-tiny2"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2026, 1, 1),
    "retries": 1,
    "retry_delay": timedelta(minutes=5)
}

dag = DAG(
    "index_documents",
    default_args=default_args,
    description="Индексация текстовых файлов в Qdrant",
    schedule_interval="@daily",
    catchup=False
)


def index_files():
    # Инициализация клиента Qdrant
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=312,
            distance=models.Distance.COSINE
        ),
    )

    # Модель
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # Сплиттер
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )

    # Чтение содержимого папки data
    data_dir = Path(DATA_PATH)
    if not data_dir.exists():
        raise FileNotFoundError(f"Папка {DATA_PATH} не найдена")

    all_chunks = []
    for file_path in data_dir.glob("*.txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        chunks = splitter.split_text(text)
        all_chunks.extend(chunks)

    if not all_chunks:
        print("Нет текстов для индексации")
        return

    # Генерирация эмбеддингов для всех чанков
    vectors = embeddings.embed_documents(all_chunks)

    # Загрузка в Qdrant
    points = []
    for idx, (chunk, vector) in enumerate(zip(all_chunks, vectors)):
        points.append(
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={"text": chunk, "source": "local_data"}
            )
        )
    client.upsert(collection_name=COLLECTION_NAME, points=points)

    print(f"Успешно проиндексировано {len(points)} чанков")


index_task = PythonOperator(
    task_id="index_files",
    python_callable=index_files,
    dag=dag
)