from typing import Any, List, Optional
from llama_index.core.base.llms.types import CompletionResponse, LLMMetadata, CompletionResponseGen
from llama_index.core.llms import CustomLLM
from llama_index.core.llms.callbacks import llm_completion_callback
from volcenginesdkarkruntime import Ark

__all__ = [
    "DoubaoLLM"
]
class DoubaoLLM(CustomLLM):

    client:Any = None
    chat_history:list = []

    context_window: int = 32768
    num_output: int = 2048
    model_name: str = "doubao-32K-pro"

    def _chat(self, user_input: Optional[str] = None) -> str:
        if user_input is None:
            return None

        if self.client is None:
            self.client = Ark(ak="******",
                     sk="******")

        completion = self.client.chat.completions.create(
            model="*****",
            messages= [{"role": "user", "content": user_input}]
        )
        content = completion.choices[0].message.content
        return content

    def _stream_chat(self, user_input: Optional[str]=None):
        if self.client is None:
            self.client = Ark(ak="******",
                     sk="*******")

        if user_input is None:
            return None

        stream = self.client.chat.completions.create(
            model="*******",
            messages=[{"role": "user", "content": user_input}],
            stream=True
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            yield chunk.choices[0].delta.content

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.model_name,
        )

    @llm_completion_callback()
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        print(prompt)
        return CompletionResponse(
            text=self._chat(user_input=prompt),
            token_usage={},
        )

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        def gen() -> CompletionResponseGen:
            for s in self._stream_chat(user_input=prompt):
                yield CompletionResponse(
                    text=s,
                    token_usage={},
                )

        print(prompt)

        return gen()

