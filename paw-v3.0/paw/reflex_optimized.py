# reflex_optimized.py
import os
import mmap
import struct
import threading
import time
from typing import Dict, List, Optional, Tuple

REC_MAGIC = b"RFLX"
REC_VER = 1
HDR = struct.Struct("<4sB7sQI")  # magic, ver, reserved, key(u64), nbytes(u32)

class ReflexStore:
    """
    Optimized append-only record store with:
    - Batch operations
    - Direct RGB addressing mode
    - Lock-free reads after finalization
    - Memory-mapped bulk writes
    """

    def __init__(
        self,
        path,
        ram_limit_bytes=256 * 1024 * 1024,
        direct_mode=False,  # Enable O(1) addressing for dense key spaces
    ):
        self.path = path
        self.direct_mode = direct_mode
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        self.fd = open(path, "a+b", buffering=0)
        self.lock = threading.Lock()
        self.index = {}  # key -> (abs_off, nbytes)
        
        self.ram = bytearray()
        self.file_size = os.path.getsize(self.path)
        self.ram_base = self.file_size
        
        self.mmap = None
        self.mmap_size = 0
        self.finalized = False  # When True, enables lock-free reads
        
        # Direct mode: assume keys are sequential starting at 0
        self.record_size = None  # Will be set on first write
        
        if self.file_size > 0:
            self._refresh_mmap()
            if not direct_mode:
                self._rebuild_index_from_mmap()
            else:
                self._detect_record_size()

    # -------------------------
    # Batch Operations
    # -------------------------

    def put_batch(self, items: Dict[int, bytes]):
        """Write multiple records with single lock acquisition"""
        if self.finalized:
            raise RuntimeError("Cannot write to finalized store")
            
        with self.lock:
            for key, payload in items.items():
                key &= 0xFFFFFFFFFFFFFFFF
                hdr = HDR.pack(REC_MAGIC, REC_VER, b"\x00" * 7, key, len(payload))
                abs_off = self.ram_base + len(self.ram)
                self.ram += hdr + payload
                
                if not self.direct_mode:
                    self.index[key] = (abs_off, len(payload))
                elif self.record_size is None:
                    self.record_size = HDR.size + len(payload)

    def get_batch(self, keys: List[int]) -> Dict[int, Optional[bytes]]:
        """Read multiple records efficiently"""
        results = {}
        
        if self.finalized:
            # Lock-free reads from mmap
            for key in keys:
                results[key] = self._get_from_mmap_unsafe(key)
        else:
            with self.lock:
                for key in keys:
                    results[key] = self._get_locked(key)
        
        return results

    # -------------------------
    # Single Operations
    # -------------------------

    def put(self, key: int, payload: bytes):
        """Write single record"""
        self.put_batch({key: payload})

    def get(self, key: int) -> Optional[bytes]:
        """Read single record"""
        key &= 0xFFFFFFFFFFFFFFFF
        
        if self.finalized:
            return self._get_from_mmap_unsafe(key)
        
        with self.lock:
            return self._get_locked(key)

    # -------------------------
    # Performance Modes
    # -------------------------

    def finalize(self):
        """
        Mark store as immutable, enabling lock-free concurrent reads.
        Call after all writes are complete.
        """
        self.flush(block=True)
        self.finalized = True

    def flush(self, block=False):
        """Write RAM buffer to disk"""
        with self.lock:
            if not self.ram:
                return

            self.fd.write(self.ram)
            self.fd.flush()
            os.fsync(self.fd.fileno())

            self.file_size += len(self.ram)
            self.ram.clear()
            self.ram_base = self.file_size

        self._refresh_mmap()

    def close(self):
        """Flush and close file handles"""
        if not self.finalized:
            self.flush(block=True)
        if self.mmap:
            try:
                self.mmap.close()
            except Exception:
                pass
            self.mmap = None
            self.mmap_size = 0
        self.fd.close()

    # -------------------------
    # Internal Helpers
    # -------------------------

    def _get_locked(self, key: int) -> Optional[bytes]:
        """Read with lock held (for non-finalized store)"""
        key &= 0xFFFFFFFFFFFFFFFF
        
        if self.direct_mode and self.record_size:
            return self._get_direct(key)
        
        entry = self.index.get(key)
        if entry is None:
            return None

        abs_off, n = entry

        # Check RAM buffer first
        if abs_off >= self.ram_base:
            rel = abs_off - self.ram_base
            if rel + HDR.size + n <= len(self.ram):
                hdr = self.ram[rel : rel + HDR.size]
                magic, ver, _, k2, n2 = HDR.unpack(hdr)
                if magic != REC_MAGIC or ver != REC_VER or k2 != key or n2 != n:
                    return None
                return bytes(self.ram[rel + HDR.size : rel + HDR.size + n])
            return None

        # Read from mmap
        return self._read_from_mmap(abs_off, n, key)

    def _get_from_mmap_unsafe(self, key: int) -> Optional[bytes]:
        """Lock-free read from finalized mmap"""
        if self.direct_mode and self.record_size:
            return self._get_direct(key)
        
        entry = self.index.get(key)
        if entry is None:
            return None
        
        abs_off, n = entry
        return self._read_from_mmap(abs_off, n, key)

    def _get_direct(self, key: int) -> Optional[bytes]:
        """O(1) direct addressing for sequential keys"""
        if not self.mmap or self.record_size is None:
            return None
        
        abs_off = key * self.record_size
        if abs_off + self.record_size > self.mmap_size:
            return None
        
        hdr = self.mmap[abs_off : abs_off + HDR.size]
        magic, ver, _, k2, n = HDR.unpack(hdr)
        
        if magic != REC_MAGIC or ver != REC_VER or k2 != key:
            return None
        
        return self.mmap[abs_off + HDR.size : abs_off + HDR.size + n]

    def _read_from_mmap(self, abs_off: int, n: int, key: int) -> Optional[bytes]:
        """Read and validate record from mmap"""
        if not self.mmap:
            self._refresh_mmap()
        if not self.mmap:
            return None

        if abs_off + HDR.size + n > self.mmap_size:
            return None

        hdr = self.mmap[abs_off : abs_off + HDR.size]
        magic, ver, _, k2, n2 = HDR.unpack(hdr)
        if magic != REC_MAGIC or ver != REC_VER or k2 != key or n2 != n:
            return None

        return self.mmap[abs_off + HDR.size : abs_off + HDR.size + n]

    def _refresh_mmap(self):
        """Update memory map if file grew"""
        size = os.path.getsize(self.path)
        if size == self.mmap_size and self.mmap is not None:
            return

        if self.mmap is not None:
            try:
                self.mmap.close()
            except Exception:
                pass

        if size <= 0:
            self.mmap = None
            self.mmap_size = 0
            return

        self.mmap = mmap.mmap(self.fd.fileno(), 0, access=mmap.ACCESS_READ)
        self.mmap_size = size

    def _rebuild_index_from_mmap(self):
        """Scan file and rebuild index"""
        if not self.mmap or self.mmap_size < HDR.size:
            return

        off = 0
        size = self.mmap_size

        while off + HDR.size <= size:
            hdr = self.mmap[off : off + HDR.size]
            try:
                magic, ver, _, key, n = HDR.unpack(hdr)
            except Exception:
                break

            if magic != REC_MAGIC or ver != REC_VER:
                break

            rec_end = off + HDR.size + n
            if rec_end > size:
                break

            self.index[key] = (off, n)
            off = rec_end

    def _detect_record_size(self):
        """Detect fixed record size in direct mode"""
        if not self.mmap or self.mmap_size < HDR.size:
            return
        
        hdr = self.mmap[0 : HDR.size]
        try:
            magic, ver, _, key, n = HDR.unpack(hdr)
            if magic == REC_MAGIC and ver == REC_VER:
                self.record_size = HDR.size + n
        except Exception:
            pass
