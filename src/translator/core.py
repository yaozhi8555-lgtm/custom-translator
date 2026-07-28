from .config import OPENROUTER_API_KEY, DEFAULT_MODEL
from .providers.openrouter import OpenRouterProvider

SYSTEM_PROMPT_QUICK = """你是一个专业翻译助手。
请将用户输入的文本准确翻译成{target_lang}。
只输出译文本身，不要添加任何解释、说明或额外内容。"""


def translate(text: str, target_lang: str = "英文", model: str = DEFAULT_MODEL) -> dict:
    provider = OpenRouterProvider(api_key=OPENROUTER_API_KEY)
    system_prompt = SYSTEM_PROMPT_QUICK.format(target_lang=target_lang)
    result = provider.chat(system_prompt=system_prompt, user_message=text, model=model)
    return result