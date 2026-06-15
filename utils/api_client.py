from openai import AsyncOpenAI
from config.settings import settings

def get_client() -> AsyncOpenAI:
    return AsyncOpenAI(
        base_url=settings.base_url,
        api_key=settings.api_key,
    )
