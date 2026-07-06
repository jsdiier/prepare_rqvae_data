import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.utils.data as data
from tqdm import tqdm

EMB_COLUMN_KEYWORDS = ["emb", "embedding", "vector", "feat"]


class EmbDataset(data.Dataset):

    def __init__(self, data_path, emb_column=None):
        """
        data_path: 支持 .npy / .npz / .parquet / .csv
        emb_column: 当输入是 parquet/csv 且存在多列时，显式指定 embedding 所在的列名

        parquet 采用流式读取：逐 row-group 转换并写入磁盘 memmap 缓存
        （<data_path>.embcache.npy），训练时按需分页加载，峰值内存与数据总量无关；
        缓存存在且比 parquet 新时直接复用。
        """
        self.data_path = data_path
        self.embeddings = self._load(data_path, emb_column)

        print(f"Loaded embeddings shape: {self.embeddings.shape}")
        self.dim = self.embeddings.shape[-1]

    def _load(self, data_path, emb_column):
        ext = os.path.splitext(data_path)[-1].lower()

        if ext == ".npy":
            arr = np.load(data_path, allow_pickle=True)
            return self._sanitize(self._to_float_array(arr))

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
            return self._sanitize(self._to_float_array(arr))

        elif ext == ".parquet":
            return self._load_parquet_streaming(data_path, emb_column)

        elif ext == ".csv":
            df = pd.read_csv(data_path)
            return self._sanitize(self._df_to_array(df, emb_column))

        else:
            raise ValueError(f"不支持的文件格式: {ext}，目前支持 .npy/.npz/.parquet/.csv")

    # ------------------------------------------------------------------
    # parquet 流式加载 + memmap 缓存
    # ------------------------------------------------------------------
    def _load_parquet_streaming(self, data_path, emb_column):
        # EMB_CACHE_DTYPE=fp16 时缓存存 float16，内存/磁盘减半；
        # __getitem__ 会转回 float32，训练侧无感知
        dtype_env = os.environ.get("EMB_CACHE_DTYPE", "fp32").lower()
        if dtype_env in ("fp16", "f16", "float16", "half"):
            self._cache_descr, self._cache_dtype = "<f2", np.float16
            suffix = ".embcache.f16.npy"
        else:
            self._cache_descr, self._cache_dtype = "<f4", np.float32
            suffix = ".embcache.npy"


        # EMB_CACHE_DIR 可把缓存重定向到本地盘（NFS 上 mmap 随机读极慢时用）
        cache_dir = os.environ.get("EMB_CACHE_DIR")
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            cache_path = os.path.join(
                cache_dir, os.path.basename(data_path) + suffix
            )
        else:
            cache_path = data_path + suffix
        pf = pq.ParquetFile(data_path)
        num_rows = pf.metadata.num_rows

        # 缓存有效：比 parquet 新且行数一致，直接复用
        if os.path.exists(cache_path) and \
                os.path.getmtime(cache_path) >= os.path.getmtime(data_path):
            cached = np.load(cache_path, mmap_mode="r")
            if len(cached) == num_rows:
                print(f"复用缓存: {cache_path} (shape={cached.shape})")
                del cached
                return self._open_cache(cache_path)
            print(f"缓存行数({len(cached)})与 parquet({num_rows})不一致，重建缓存")
            del cached

        emb_column = self._resolve_emb_column(pf.schema_arrow.names, emb_column)
        print(f"流式读取 parquet: {data_path} (rows={num_rows}, column='{emb_column}')")

        tmp_path = cache_path + ".tmp"
        out = None
        dim = None
        offset = 0
        nan_cnt = 0
        inf_cnt = 0
        # 逐 batch 读取：任意时刻内存中只有一个 batch。
        # 采用顺序 append 写普通 .npy 文件（先写 header 再逐块追加数据），
        # 不用 memmap 原地填：NFS 上 mmap 脏页回写极慢，顺序 write 快一个量级，
        # 且文件随写入增长，du 可直接当进度参考。
        # 全部完成后原子重命名，避免中途被 kill 留下半成品被误复用。
        pbar = tqdm(total=num_rows, desc="parquet -> npy cache", unit="row", ncols=100)
        try:
            for batch in pf.iter_batches(batch_size=65536, columns=[emb_column]):
                chunk = self._arrow_list_to_matrix(batch.column(0))

                if out is None:
                    dim = chunk.shape[1]
                    out = open(tmp_path, "wb")
                    # write_array_header_1_0 自带 magic 前缀，不要重复写
                    np.lib.format.write_array_header_1_0(
                        out,
                        {"descr": self._cache_descr, "fortran_order": False,
                         "shape": (num_rows, dim)},
                    )
                elif chunk.shape[1] != dim:
                    raise ValueError(
                        f"embedding 维度不一致: 之前为 {dim}，当前 batch 为 {chunk.shape[1]}"
                    )

                nan_mask = np.isnan(chunk)
                if nan_mask.any():
                    nan_cnt += int(nan_mask.sum())
                    chunk[nan_mask] = 0.0
                inf_mask = np.isinf(chunk)
                if inf_mask.any():
                    inf_cnt += int(inf_mask.sum())
                    chunk[inf_mask] = 0.0

                out.write(np.ascontiguousarray(chunk, dtype=self._cache_dtype).tobytes())
                offset += len(chunk)
                pbar.update(len(chunk))
        finally:
            pbar.close()
            if out is not None:
                out.close()

        if out is None:
            raise ValueError(f"parquet 文件为空: {data_path}")
        if offset != num_rows:
            raise ValueError(f"实际读取行数({offset})与元数据行数({num_rows})不一致")

        if nan_cnt:
            print(f"Warning: Found {nan_cnt} NaN values in embeddings (已置 0)")
        if inf_cnt:
            print(f"Warning: Found {inf_cnt} infinite values in embeddings (已置 0)")

        os.replace(tmp_path, cache_path)
        print(f"npy 缓存已写入: {cache_path}")
        return self._open_cache(cache_path)

    def _open_cache(self, cache_path):
        """
        打开缓存：内存装得下就整体读进 RAM（顺序读，NFS 友好，训练全速）；
        装不下退化为只读 mmap 按需分页（本地 SSD 上没问题，NFS 上随机读会慢）。
        可用环境变量 EMB_CACHE_IN_MEMORY=1/0 强制开启/关闭整载。
        """
        size = os.path.getsize(cache_path)
        mode = os.environ.get("EMB_CACHE_IN_MEMORY", "auto").lower()
        if mode in ("1", "true", "yes"):
            in_memory = True
        elif mode in ("0", "false", "no"):
            in_memory = False
        else:
            avail = self._available_memory_bytes()
            in_memory = avail is not None and avail > size * 1.2

        if in_memory:
            print(f"整体加载缓存到内存: {size / 2**30:.1f} GB，顺序读取中…")
            arr = np.load(cache_path)
            print("加载完成")
            return arr

        print("以只读 mmap 打开缓存（按需分页；如内存充足可设 EMB_CACHE_IN_MEMORY=1 整载提速）")
        return np.load(cache_path, mmap_mode="r")

    @staticmethod
    def _available_memory_bytes():
        avail = None
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        avail = int(line.split()[1]) * 1024
                        break
        except OSError:
            return None

        # 容器内 /proc/meminfo 显示的是宿主机内存，真正的约束是 cgroup 限额，
        # 取两者较小值，避免整载时撞限额被 OOM kill
        for limit_path, usage_path in (
            ("/sys/fs/cgroup/memory.max",
             "/sys/fs/cgroup/memory.current"),                      # cgroup v2
            ("/sys/fs/cgroup/memory/memory.limit_in_bytes",
             "/sys/fs/cgroup/memory/memory.usage_in_bytes"),        # cgroup v1
        ):
            try:
                with open(limit_path) as f:
                    raw = f.read().strip()
                if raw == "max":
                    break
                limit = int(raw)
                if limit >= 1 << 60:  # v1 无限额时是个近似无穷大的数
                    break
                with open(usage_path) as f:
                    usage = int(f.read().strip())
                cg_avail = max(limit - usage, 0)
                avail = cg_avail if avail is None else min(avail, cg_avail)
                break
            except (OSError, ValueError):
                continue

        # k8s 的限额常挂在 Pod 层（容器自身 cgroup 显示无限额），
        # 该层限额可通过 memory.stat 的 hierarchical_memory_limit 看到
        try:
            hier_limit = None
            with open("/sys/fs/cgroup/memory/memory.stat") as f:
                for line in f:
                    if line.startswith("hierarchical_memory_limit"):
                        hier_limit = int(line.split()[1])
                        break
            if hier_limit is not None and hier_limit < 1 << 60:
                with open("/sys/fs/cgroup/memory/memory.usage_in_bytes") as f:
                    usage = int(f.read().strip())
                cg_avail = max(hier_limit - usage, 0)
                avail = cg_avail if avail is None else min(avail, cg_avail)
        except (OSError, ValueError):
            pass
        return avail

    def _resolve_emb_column(self, column_names, emb_column):
        if emb_column is not None:
            if emb_column not in column_names:
                raise ValueError(
                    f"指定的列 '{emb_column}' 不在文件中，现有列: {list(column_names)}"
                )
            return emb_column

        if len(column_names) == 1:
            return column_names[0]

        candidates = [c for c in column_names
                      if any(k in c.lower() for k in EMB_COLUMN_KEYWORDS)]
        if len(candidates) == 1:
            print(f"自动识别 embedding 列: '{candidates[0]}'")
            return candidates[0]

        raise ValueError(
            f"无法自动确定 embedding 列，现有列: {list(column_names)}，"
            f"请通过 emb_column 参数显式指定"
        )

    @staticmethod
    def _arrow_list_to_matrix(col):
        """arrow list<float> / fixed_size_list<float> 列 → (n, dim) float32 矩阵"""
        if col.null_count:
            raise ValueError(f"embedding 列存在 {col.null_count} 个 null 值，请先清洗数据")

        flat = col.flatten()
        values = flat.to_numpy(zero_copy_only=False)
        n = len(col)
        if n == 0 or len(values) % n != 0:
            raise ValueError(
                f"embedding 列长度不齐: {n} 行共 {len(values)} 个元素，无法 reshape 成矩阵"
            )
        matrix = np.ascontiguousarray(
            values.reshape(n, len(values) // n), dtype=np.float32
        )
        if not matrix.flags.writeable:
            # arrow 零拷贝出来的 buffer 只读，NaN/inf 清洗需要可写副本
            matrix = matrix.copy()
        return matrix

    # ------------------------------------------------------------------
    # csv / 小数据路径（保持原逻辑）
    # ------------------------------------------------------------------
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
                      if any(k in c.lower() for k in EMB_COLUMN_KEYWORDS)]
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

    @staticmethod
    def _sanitize(arr):
        nan_mask = np.isnan(arr)
        if nan_mask.any():
            print(f"Warning: Found {nan_mask.sum()} NaN values in embeddings")
            arr[nan_mask] = 0.0
        inf_mask = np.isinf(arr)
        if inf_mask.any():
            print(f"Warning: Found {inf_mask.sum()} infinite values in embeddings")
            arr[inf_mask] = 0.0
        return arr

    def __getitem__(self, index):
        # memmap 场景下先拷贝出本条数据，避免 torch 直接引用只读缓冲区
        emb = np.array(self.embeddings[index], dtype=np.float32)
        return torch.from_numpy(emb)

    def __len__(self):
        return len(self.embeddings)