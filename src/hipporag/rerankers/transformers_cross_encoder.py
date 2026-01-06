from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


@dataclass
class RerankResult:
    scores: List[float]
    sorted_indices: List[int]


class TransformersCrossEncoderReranker:
    """
    本地 Cross-Encoder rerank（Transformers）。

    典型模型：
    - BAAI/bge-reranker-v2-m3（多语种/中文友好，效果强）
    - BAAI/bge-reranker-large / BAAI/bge-reranker-base（英文更常见，体量更小）
    - jinaai/jina-reranker-v2-base-multilingual（多语种）
    - mixedbread-ai/mxbai-rerank-large-v1（英文强）
    """

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        batch_size: int = 32,
        max_length: int = 512,
        use_fp16: bool = True,
    ):
        self.model_name = model_name
        self.device = self._resolve_device(device)
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        self.use_fp16 = bool(use_fp16)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, trust_remote_code=True,device_map="auto")
        # self.model.to(self.device)
        self.model.eval()

    def _resolve_device(self, device: str) -> torch.device:
        if device is None or device == "auto":
            return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        return torch.device(device)

    def rerank(self, query: str, docs: Sequence[str], top_k: Optional[int] = None) -> RerankResult:
        if not docs:
            return RerankResult(scores=[], sorted_indices=[])

        # 逐批打分
        scores: List[float] = [0.0] * len(docs)
        use_autocast = self.use_fp16 and self.device.type == "cuda"

        with torch.inference_mode():
            for start in range(0, len(docs), self.batch_size):
                batch_docs = list(docs[start : start + self.batch_size])
                encoded = self.tokenizer(
                    [query] * len(batch_docs),
                    batch_docs,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {k: v.to(self.device) for k, v in encoded.items()}

                if use_autocast:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        out = self.model(**encoded)
                else:
                    out = self.model(**encoded)

                logits = out.logits
                if logits.ndim == 2 and logits.shape[-1] == 1:
                    batch_scores = logits.squeeze(-1)
                else:
                    # 多分类模型取“相关”维度并不统一，保守起见取最后一维
                    batch_scores = logits[..., -1]

                batch_scores = batch_scores.detach().float().cpu().tolist()
                for i, s in enumerate(batch_scores):
                    scores[start + i] = float(s)

        sorted_indices = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)
        if top_k is not None:
            sorted_indices = sorted_indices[: int(top_k)]
        return RerankResult(scores=scores, sorted_indices=sorted_indices)



