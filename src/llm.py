from functools import cache

from langsmith import traceable
from langsmith.wrappers import wrap_openai
from openai import OpenAI

from .config import setting
from .prompts import ANSWER_PROMPT, SYSTEM_PROMPT

LLM_MODEL = "gpt-4o-mini"


@cache
def _client() -> OpenAI:
    return wrap_openai(OpenAI(api_key=setting.OPENAI_API_KEY, max_retries=10))


@traceable(name="Generate Answer")
def generate_response_from_context(query: str, context: list[str]) -> str:
    completion = _client().chat.completions.create(
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
