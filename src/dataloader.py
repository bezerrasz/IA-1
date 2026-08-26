"""Construção de lotes de dados para o pipeline da Sprint 2."""

from __future__ import annotations

from torch.utils.data import DataLoader

try:
    from .config import DEFAULT_CONFIG
    from .dataset import Encoder, GPTDatasetV1
except ImportError:  # Permite importar o módulo diretamente a partir de src/.
    from config import DEFAULT_CONFIG
    from dataset import Encoder, GPTDatasetV1


def create_dataloader_v1(
    txt: str,
    tokenizer: Encoder,
    batch_size: int | None = None,
    max_length: int | None = None,
    stride: int | None = None,
    shuffle: bool | None = None,
    drop_last: bool | None = None,
    num_workers: int = 0,
) -> DataLoader:
    """Cria um DataLoader com os pares de entrada e alvo do corpus."""

    config = DEFAULT_CONFIG
    resolved_batch_size = config.batch_size if batch_size is None else batch_size
    resolved_max_length = config.context_length if max_length is None else max_length
    resolved_stride = config.stride if stride is None else stride
    resolved_shuffle = config.shuffle if shuffle is None else shuffle
    resolved_drop_last = config.drop_last if drop_last is None else drop_last

    if resolved_batch_size <= 0:
        raise ValueError("batch_size deve ser maior que zero")
    if num_workers < 0:
        raise ValueError("num_workers não pode ser negativo")

    dataset = GPTDatasetV1(
        txt,
        tokenizer,
        max_length=resolved_max_length,
        stride=resolved_stride,
    )
    return DataLoader(
        dataset,
        batch_size=resolved_batch_size,
        shuffle=resolved_shuffle,
        drop_last=resolved_drop_last,
        num_workers=num_workers,
    )


__all__ = ["create_dataloader_v1"]
