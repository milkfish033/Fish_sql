from typing import List, Any
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from llama_index.core.embeddings import BaseEmbedding

import logging

logging.basicConfig(filemode="w", format="%(asctime)s %(name)s:%(levelname)s:%(message)s", datefmt="%d-%M-%Y %H:%M:%S", level=logging.INFO)


class LocalGTEBaseEmbedding(BaseEmbedding):
    # model = "iic/nlp_gte_sentence-embedding_chinese-base"
    # resource: https://modelscope.cn/models/iic/nlp_gte_sentence-embedding_chinese-base
    # sequence_length = 512
    _modelscope = pipeline(Tasks.sentence_embedding,
                                 model="iic/nlp_gte_sentence-embedding_chinese-base",
                                 sequence_length=512)

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
    def _extract_embed(self, raw: dict) -> List[float]:
        if raw is None:
            return None

        text_embedding = raw.get('text_embedding')
        if text_embedding is not None:
            return text_embedding[0].tolist()
        return None

    def _get_text_embedding(self, text: str) -> List[float]:
        embedding = self._modelscope({
            "source_sentence": [text]
        })
        return self._extract_embed(embedding)

    def _get_query_embedding(self, query: str) -> List[float]:
        embedding = self._modelscope({
            "source_sentence": [query]
        })
        return self._extract_embed(embedding)

    # def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
    #     embeddings = [self._get_text_embedding(i) for i in texts]
    #     return embeddings

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)


if __name__ == '__main__':
    embed = LocalGTEBaseEmbedding()

    res = embed.get_query_embedding("Hello")

    print(len(res))

    
