from llm_base import LLMBase

class Assistant(LLMBase):
    def __init__(self):
        super().__init__()


    def ask(self,prompt: str,system="",schema=None):
        res = self.call_structured_json(
            model=self.STRONG_TEXT_MODEL,
            system_prompt=system,
            user_prompt=prompt,
            json_schema=schema,
            temperature=0.7,
        )
        return res

