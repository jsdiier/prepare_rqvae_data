import os
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data


class EmbDataset(data.Dataset):

    def __init__(self, data_path, emb_column=None):
        """
        data_path: 支持 .npy / .npz / .parquet / .csv
        emb_column: 当输入是 parquet/csv 且存在多列时，显式指定 embedding 所在的列名
        """
        self.data_path = data_path
        self.embeddings = self._load(data_path, emb_column)

        # Check for NaN values and handle them
        nan_mask = np.isnan(self.embeddings)
        if nan_mask.any():
            print(f"Warning: Found {nan_mask.sum()} NaN values in embeddings")
            self.embeddings[nan_mask] = 0.0

        # Check for infinite values
        inf_mask = np.isinf(self.embeddings)
        if inf_mask.any():
            print(f"Warning: Found {inf_mask.sum()} infinite values in embeddings")
            self.embeddings[inf_mask] = 0.0

        print(f"Loaded embeddings shape: {self.embeddings.shape}")
        print(f"Embeddings stats - min: {self.embeddings.min():.6f}, "
              f"max: {self.embeddings.max():.6f}, mean: {self.embeddings.mean():.6f}")

        self.dim = self.embeddings.shape[-1]

    def _load(self, data_path, emb_column):
        ext = os.path.splitext(data_path)[-1].lower()

        if ext == ".npy":
            arr = np.load(data_path, allow_pickle=True)
            return self._to_float_array(arr)

        elif ext == ".npz":
            npz = np.load(data_path, allow_pickle=True)
            # 默认取第一个 key，如果只有一个数组
            keys = list(npz.keys())
            if len(keys) == 1:
                arr = npz[keys[0]]
            elif emb_column is not None and emb_column in keys:
                arr = npz[emb_column]
            else:
                raise ValueError(
                    f".npz 文件包含多个数组: {keys}，请通过 emb_column 参数指定使用哪一个"
                )
            return self._to_float_array(arr)

        elif ext == ".parquet":
            df = pd.read_parquet(data_path)
            return self._df_to_array(df, emb_column)

        elif ext == ".csv":
            df = pd.read_csv(data_path)
            return self._df_to_array(df, emb_column)

        else:
            raise ValueError(f"不支持的文件格式: {ext}，目前支持 .npy/.npz/.parquet/.csv")

    def _df_to_array(self, df, emb_column):
        # 1. 显式指定列名
        if emb_column is not None:
            if emb_column not in df.columns:
                raise ValueError(f"指定的列 '{emb_column}' 不在文件中，现有列: {list(df.columns)}")
            col = df[emb_column]
            return self._series_to_array(col)

        # 2. 只有一列，直接用
        if df.shape[1] == 1:
            return self._series_to_array(df.iloc[:, 0])

        # 3. 尝试按常见命名自动探测
        candidates = [c for c in df.columns
                      if any(k in c.lower() for k in ["emb", "embedding", "vector", "feat"])]
        if len(candidates) == 1:
            print(f"自动识别 embedding 列: '{candidates[0]}'")
            return self._series_to_array(df[candidates[0]])

        # 4. 都不满足，看看是不是所有列都是数值列（即每行是一个展开的向量，每维一列）
        if all(pd.api.types.is_numeric_dtype(df[c]) for c in df.columns):
            print(f"检测到 {df.shape[1]} 个数值列，按展开的向量维度处理")
            return df.to_numpy(dtype=np.float32)

        raise ValueError(
            f"无法自动确定 embedding 列，现有列: {list(df.columns)}，"
            f"请通过 emb_column 参数显式指定"
        )

    def _series_to_array(self, series):
        arr = np.stack(series.to_numpy())
        return self._to_float_array(arr)

    def _to_float_array(self, arr):
        # 处理 0-d object array（例如 pickle 存的 dict/list）的情况
        if arr.dtype == object and arr.shape == ():
            arr = arr.item()
            if isinstance(arr, dict):
                raise ValueError(
                    f"加载到的是 dict 而非数组，keys: {list(arr.keys())}，"
                    f"请确认数据格式或手动提取所需字段"
                )
            arr = np.array(arr)
        return np.asarray(arr, dtype=np.float32)

    def __getitem__(self, index):
        emb = self.embeddings[index]
        tensor_emb = torch.FloatTensor(emb)
        return tensor_emb

    def __len__(self):
        return len(self.embeddings)