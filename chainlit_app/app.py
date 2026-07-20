import os
import uuid
import requests
import chainlit as cl
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
import urllib3


urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = "documents"
EMBEDDING_MODEL = "cointegrated/rubert-tiny2"
TOP_K = 3
GIGACHAT_CREDENTIALS = os.getenv("GIGACHAT_CREDENTIALS")
GIGACHAT_AUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
GIGACHAT_API_URL = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"

# Инициализация клиентов (глобально)
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def get_gigachat_token():
    """Получает временный access_token по схеме OAuth 2.0 (Client Credentials)"""
    headers = {
        "Authorization": f"Basic {GIGACHAT_CREDENTIALS}",
        "RqUID": str(uuid.uuid4()),
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json"
    }
    data = {
        "scope": "GIGACHAT_API_PERS"  # физлицо
    }

    try:
        response = requests.post(GIGACHAT_AUTH_URL, headers=headers, data=data, verify=False, timeout=10)
        response.raise_for_status()
        token_data = response.json()
        return token_data["access_token"]
    except requests.exceptions.RequestException as e:
        raise Exception(f"Ошибка получения токена GigaChat: {e}") from e


@cl.on_chat_start
async def start():
    await cl.Message(content="Привет! Я RAG-ассистент. Задай любой вопрос по базе знаний.").send()


@cl.on_message
async def main(message: cl.Message):
    user_query = message.content

    # Векторизация вопроса
    query_vector = embeddings.embed_query(user_query)

    # Поиск похожих чанков для контекста
    search_result = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=TOP_K,
        with_payload=True
    ).points

    if not search_result:
        await cl.Message(content="Извините, ничего не найдено в базе знаний.").send()
        return

    context = "\n\n".join([hit.payload["text"] for hit in search_result])


    prompt = f"""
Ты — полезный ассистент. Ответь на вопрос пользователя, используя только информацию из контекста.
Если в контексте нет ответа, скажи об этом честно.

Контекст:
{context}

Вопрос пользователя: {user_query}

Ответ:
"""

    # Обращение к LLM (тут GigaChat)
    try:
        access_token = get_gigachat_token()

        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        payload = {
            "model": "GigaChat",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 500
        }

        response = requests.post(GIGACHAT_API_URL, headers=headers, json=payload, timeout=30, verify=False)
        response.raise_for_status()

        # Парсинг ответа
        answer = response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        answer = f"Ошибка при обращении к GigaChat: {e}"

    await cl.Message(content=answer).send()