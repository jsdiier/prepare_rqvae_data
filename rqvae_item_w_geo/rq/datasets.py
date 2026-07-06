import math
import os

import numpy as np
import torch
import torch.utils.data as data
import pyarrow.parquet as pq


class EmbDataset(data.Dataset):

    def __init__(self, data_path, embedding_col="embedding"):

        self.data_path = data_path

        ext = os.path.splitext(data_path)[-1].lower()
        if ext == ".parquet":
            self.embeddings = self._load_from_parquet(data_path, embedding_col)
        else:
            # 兼容旧的 .npy 格式
            self.embeddings = np.load(data_path)

        # Check for NaN values and handle them
        nan_mask = np.isnan(self.embeddings)
        if nan_mask.any():
            print(f"Warning: Found {nan_mask.sum()} NaN values in embeddings")
            # Replace NaN with zeros
            self.embeddings[nan_mask] = 0.0
            
        # Check for infinite values
        inf_mask = np.isinf(self.embeddings)
        if inf_mask.any():
            print(f"Warning: Found {inf_mask.sum()} infinite values in embeddings")
            # Replace inf with zeros
            self.embeddings[inf_mask] = 0.0
            
        print(f"Loaded embeddings shape: {self.embeddings.shape}")
        print(f"Embeddings stats - min: {self.embeddings.min():.6f}, max: {self.embeddings.max():.6f}, mean: {self.embeddings.mean():.6f}")
        
        self.dim = self.embeddings.shape[-1]

    @staticmethod
    def _load_from_parquet(data_path, embedding_col: str) -> np.ndarray:
        """
        高效读取 parquet 里的 list<float32> embedding 列，
        直接基于 Arrow 的底层 buffer reshape 成 (N, dim) 的 ndarray，
        避免走 pandas 逐行转 python list 的慢路径。
        """
        table = pq.read_table(data_path, columns=[embedding_col])
        col = table.column(embedding_col).combine_chunks()

        if len(col) == 0:
            raise ValueError(f"parquet 中没有读到任何数据: {data_path}")

        offsets = col.offsets.to_numpy()
        lengths = np.diff(offsets)
        dim = int(lengths[0])
        if not np.all(lengths == dim):
            raise ValueError(
                f"embedding 长度不一致，无法 reshape 为矩阵，"
                f"发现长度集合: {sorted(set(lengths.tolist()))}"
            )

        values = col.values.to_numpy(zero_copy_only=False)
        embeddings = values.reshape(len(col), dim).astype(np.float32)
        return embeddings

    def __getitem__(self, index):
        emb = self.embeddings[index]
        tensor_emb = torch.FloatTensor(emb)
        return tensor_emb

    def __len__(self):
        return len(self.embeddings)


class StreamingEmbBatchDataset(data.IterableDataset):

    def __init__(self, data_path, batch_size, embedding_col="embedding"):
        self.data_path = data_path
        self.batch_size = batch_size
        self.embedding_col = embedding_col

        ext = os.path.splitext(data_path)[-1].lower()
        self.is_parquet = ext == ".parquet"

        if self.is_parquet:
            parquet_file = pq.ParquetFile(data_path)
            self.num_rows = parquet_file.metadata.num_rows
            if self.num_rows == 0:
                raise ValueError(f"parquet 中没有读到任何数据: {data_path}")

            first_batch = next(parquet_file.iter_batches(batch_size=1, columns=[embedding_col]))
            first_embedding = first_batch.column(0)[0].as_py()
            self.dim = len(first_embedding)
        else:
            # mmap 模式不会一次性把整个 npy 加载到内存里
            self.embeddings = np.load(data_path, mmap_mode="r")
            self.num_rows = len(self.embeddings)
            if self.num_rows == 0:
                raise ValueError(f"npy 中没有读到任何数据: {data_path}")
            self.dim = self.embeddings.shape[-1]

        self.num_batches = math.ceil(self.num_rows / self.batch_size)

    @staticmethod
    def _sanitize_embeddings(embeddings: np.ndarray) -> np.ndarray:
        embeddings = np.array(embeddings, dtype=np.float32, copy=True, order="C")

        nan_count = int(np.isnan(embeddings).sum())
        inf_count = int(np.isinf(embeddings).sum())
        if nan_count or inf_count:
            print(
                f"Warning: Found {nan_count} NaN values and {inf_count} infinite values in current batch"
            )
            np.nan_to_num(embeddings, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        return embeddings

    def _iter_parquet_batches(self):
        parquet_file = pq.ParquetFile(self.data_path)
        for batch in parquet_file.iter_batches(batch_size=self.batch_size, columns=[self.embedding_col]):
            col = batch.column(0)
            offsets = col.offsets.to_numpy()
            lengths = np.diff(offsets)
            dim = int(lengths[0])
            if not np.all(lengths == dim):
                raise ValueError(
                    f"embedding 长度不一致，无法 reshape 为矩阵，"
                    f"发现长度集合: {sorted(set(lengths.tolist()))}"
                )

            values = col.values.to_numpy(zero_copy_only=False)
            embeddings = values.reshape(len(col), dim)
            embeddings = self._sanitize_embeddings(embeddings)
            yield torch.from_numpy(embeddings)

    def _iter_npy_batches(self):
        for start in range(0, self.num_rows, self.batch_size):
            embeddings = np.asarray(self.embeddings[start:start + self.batch_size])
            embeddings = self._sanitize_embeddings(embeddings)
            yield torch.from_numpy(embeddings)

    def __iter__(self):
        worker_info = data.get_worker_info()
        if worker_info is not None and worker_info.num_workers > 1:
            raise RuntimeError("StreamingEmbBatchDataset 仅支持单 worker 流式读取")

        if self.is_parquet:
            yield from self._iter_parquet_batches()
        else:
            yield from self._iter_npy_batches()

    def __len__(self):
        return self.num_batches
