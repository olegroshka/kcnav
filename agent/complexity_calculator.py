# agent/complexity_calculator.py
import zlib
import bz2
import lzma
from typing import Optional


class NcdCalculator:
    """
    Calculates Normalized Compression Distance (NCD) using standard compressors.
    Provides methods for local and global complexity measures based on NCD.
    """
    SUPPORTED_COMPRESSORS = {'zlib', 'bz2', 'lzma'}

    def __init__(self, compressor_name: str = 'zlib'):
        if compressor_name not in self.SUPPORTED_COMPRESSORS:
            raise ValueError(f"Unsupported compressor: {compressor_name}. "
                             f"Supported are: {self.SUPPORTED_COMPRESSORS}")
        self.compressor_name = compressor_name
        print(f"NCD Calculator: Using '{self.compressor_name}' compressor.")

    def _compress(self, text: str) -> bytes:
        if not isinstance(text, str):
            text = str(text)  # Ensure input is string
        data = text.encode('utf-8', errors='replace')
        if self.compressor_name == 'zlib':
            return zlib.compress(data)
        elif self.compressor_name == 'bz2':
            return bz2.compress(data)
        elif self.compressor_name == 'lzma':
            return lzma.compress(data)
        else:
            # Should not happen due to __init__ check
            raise ValueError("Internal error: Compressor configuration issue.")

    def get_compressed_length(self, text: str) -> int:
        if not text:
            return 0
        return len(self._compress(text))

    def calculate_ncd(self, text1: Optional[str], text2: Optional[str]) -> float:
        str1 = text1 or ""
        str2 = text2 or ""

        if not str1 and not str2:
            return 0.0

        c_x = self.get_compressed_length(str1)
        c_y = self.get_compressed_length(str2)

        combined_text = str1 + str2  # Ensure order for C(xy) is consistent if xy vs yx matters
        c_xy = self.get_compressed_length(combined_text)

        min_c = min(c_x, c_y)
        max_c = max(c_x, c_y)

        if max_c == 0:
            return 0.0  # Avoid division by zero

        ncd = (c_xy - min_c) / max_c
        return ncd

    def calculate_local_ncd(self, prev_step_content: Optional[str], current_step_content: str) -> float:
        return self.calculate_ncd(prev_step_content, current_step_content)

    def calculate_global_complexity(self, full_history_text: str) -> int:
        # Global complexity as compressed length of the entire history so far.
        return self.get_compressed_length(full_history_text)
