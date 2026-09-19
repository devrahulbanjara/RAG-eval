from groq import Groq

from .config import setting
from .prompts import ANSWER_PROMPT, SYSTEM_PROMPT

LLM_MODEL = "openai/gpt-oss-120b"


def generate_response_from_context(query: str, context: list[str]) -> str:
    client = Groq(api_key=setting.GROQ_API_KEY)
    completion = client.chat.completions.create(
        model=LLM_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": ANSWER_PROMPT.format(
                    context="\n\n".join(context), question=query
                ),
            },
        ],
    )
    return completion.choices[0].message.content
