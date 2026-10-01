def encode_rad(input_file: str, output_dir: str) -> None: ...

from collections.abc import Buffer
from typing import ClassVar

class SplatInput:
    """RGB-major sh_feature shape (N,3,C), where C=1,4,9,16 determines SH degree."""
    def __init__(self, position: Buffer, rotation: Buffer, log_scaling: Buffer,
                 alpha_logit: Buffer, sh_feature: Buffer, labels: Buffer) -> None: ...

class MergedLevel:
    def __init__(self, position: Buffer, rotation: Buffer, scales: Buffer,
                 opacity: Buffer, labels: Buffer, right_weight: Buffer) -> None: ...

class Compression:
    Gz: ClassVar[Compression]
    Zstd: ClassVar[Compression]

class EncodeTimings:
    @property
    def input_seconds(self) -> float: ...
    @property
    def assemble_seconds(self) -> float: ...
    @property
    def prune_seconds(self) -> float: ...
    @property
    def chunk_seconds(self) -> float: ...
    @property
    def resolve_seconds(self) -> float: ...
    @property
    def encode_seconds(self) -> float: ...
    @property
    def write_seconds(self) -> float: ...

def encode_merged_arrays(leaves: SplatInput, parents: MergedLevel, children: Buffer,
                         output_dir: str, lod_base: float,
                         compression: Compression = Compression.Gz) -> None: ...

def encode_merged_archive(leaves: SplatInput, parents: MergedLevel, children: Buffer,
                          output_file: str, lod_base: float,
                          compression: Compression = Compression.Gz) -> EncodeTimings: ...
