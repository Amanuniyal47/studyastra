"""Hardware and memory profiling utilities for Stage 11.1 benchmarking."""

import os
import sys
import platform
import time
import tracemalloc
import ctypes
from dataclasses import dataclass, asdict, field
from typing import Dict, Any, Optional, Tuple, List
import urllib.request
import json

# Windows Memory API Definitions via standard ctypes
_IS_WINDOWS = sys.platform.startswith("win")
if _IS_WINDOWS:
    from ctypes import wintypes

    class _PROCESS_MEMORY_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    class _MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", wintypes.DWORD),
            ("dwMemoryLoad", wintypes.DWORD),
            ("ullTotalPhys", ctypes.c_uint64),
            ("ullAvailPhys", ctypes.c_uint64),
            ("ullTotalPageFile", ctypes.c_uint64),
            ("ullAvailPageFile", ctypes.c_uint64),
            ("ullTotalVirtual", ctypes.c_uint64),
            ("ullAvailVirtual", ctypes.c_uint64),
            ("sullAvailExtendedVirtual", ctypes.c_uint64),
        ]


@dataclass
class ProcessMemorySnapshot:
    """Snapshot of process memory usage across heap, RSS, and optional GPU VRAM."""
    heap_current_mb: float = 0.0
    heap_peak_mb: float = 0.0
    rss_mb: float = 0.0
    peak_rss_mb: float = 0.0
    gpu_vram_mb: Optional[float] = None
    gpu_vram_peak_mb: Optional[float] = None


def get_current_process_memory() -> Tuple[float, float]:
    """Return (current_rss_mb, peak_rss_mb) for the current process."""
    if _IS_WINDOWS:
        try:
            psapi = ctypes.WinDLL("psapi")
            kernel32 = ctypes.WinDLL("kernel32")
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
            psapi.GetProcessMemoryInfo.restype = wintypes.BOOL

            pmc = _PROCESS_MEMORY_COUNTERS()
            pmc.cb = ctypes.sizeof(_PROCESS_MEMORY_COUNTERS)
            handle = kernel32.GetCurrentProcess()
            if psapi.GetProcessMemoryInfo(handle, ctypes.byref(pmc), pmc.cb):
                rss_mb = pmc.WorkingSetSize / (1024 * 1024)
                peak_rss_mb = pmc.PeakWorkingSetSize / (1024 * 1024)
                return round(rss_mb, 2), round(peak_rss_mb, 2)
        except Exception:
            pass
    else:
        try:
            import resource
            usage = resource.getrusage(resource.RUSAGE_SELF)
            # On Linux ru_maxrss is in KB; on macOS it is in bytes
            multiplier = 1.0 / 1024 if sys.platform.startswith("linux") else 1.0 / (1024 * 1024)
            peak_mb = usage.ru_maxrss * multiplier
            return round(peak_mb, 2), round(peak_mb, 2)
        except Exception:
            pass

    return 0.0, 0.0


def get_gpu_memory() -> Tuple[Optional[float], Optional[float]]:
    """Return (allocated_vram_mb, peak_vram_mb) if torch.cuda is available."""
    try:
        import torch
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / (1024 * 1024)
            peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
            return round(alloc, 2), round(peak, 2)
    except Exception:
        pass
    return None, None


class MemoryTracker:
    """Context manager for profiling memory delta across heap, RSS, and GPU."""

    def __init__(self, track_heap: bool = True):
        self.track_heap = track_heap
        self.start_snapshot = ProcessMemorySnapshot()
        self.end_snapshot = ProcessMemorySnapshot()
        self.delta_heap_peak_mb = 0.0
        self.delta_rss_mb = 0.0
        self.delta_vram_mb: Optional[float] = None
        self._was_tracemalloc_running = False

    def __enter__(self):
        if self.track_heap:
            self._was_tracemalloc_running = tracemalloc.is_tracing()
            if not self._was_tracemalloc_running:
                tracemalloc.start()
            tracemalloc.reset_peak()

        rss, peak_rss = get_current_process_memory()
        vram, peak_vram = get_gpu_memory()
        h_cur, h_peak = (0.0, 0.0)
        if self.track_heap and tracemalloc.is_tracing():
            c, p = tracemalloc.get_traced_memory()
            h_cur, h_peak = c / (1024 * 1024), p / (1024 * 1024)

        self.start_snapshot = ProcessMemorySnapshot(
            heap_current_mb=h_cur,
            heap_peak_mb=h_peak,
            rss_mb=rss,
            peak_rss_mb=peak_rss,
            gpu_vram_mb=vram,
            gpu_vram_peak_mb=peak_vram,
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        rss, peak_rss = get_current_process_memory()
        vram, peak_vram = get_gpu_memory()
        h_cur, h_peak = (0.0, 0.0)
        if self.track_heap and tracemalloc.is_tracing():
            c, p = tracemalloc.get_traced_memory()
            h_cur, h_peak = c / (1024 * 1024), p / (1024 * 1024)
            if not self._was_tracemalloc_running:
                tracemalloc.stop()

        self.end_snapshot = ProcessMemorySnapshot(
            heap_current_mb=h_cur,
            heap_peak_mb=h_peak,
            rss_mb=rss,
            peak_rss_mb=peak_rss,
            gpu_vram_mb=vram,
            gpu_vram_peak_mb=peak_vram,
        )

        self.delta_heap_peak_mb = round(self.end_snapshot.heap_peak_mb, 3)
        self.delta_rss_mb = round(max(0.0, self.end_snapshot.peak_rss_mb - self.start_snapshot.rss_mb), 3)
        if self.end_snapshot.gpu_vram_mb is not None and self.start_snapshot.gpu_vram_mb is not None:
            self.delta_vram_mb = round(self.end_snapshot.gpu_vram_mb - self.start_snapshot.gpu_vram_mb, 3)


class HardwareProfiler:
    """Collects operating system, CPU, RAM, and GPU hardware metadata."""

    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "os_name": platform.system(),
            "os_release": platform.release(),
            "os_version": platform.version(),
            "architecture": platform.machine(),
            "python_version": platform.python_version(),
            "python_compiler": platform.python_compiler(),
            "cpu_count_logical": os.cpu_count() or 0,
        }

        # Physical RAM on Windows
        if _IS_WINDOWS:
            try:
                stat = _MEMORYSTATUSEX()
                stat.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
                if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                    info["total_ram_gb"] = round(stat.ullTotalPhys / (1024**3), 2)
                    info["available_ram_gb"] = round(stat.ullAvailPhys / (1024**3), 2)
            except Exception:
                info["total_ram_gb"] = "Unknown"
                info["available_ram_gb"] = "Unknown"
        else:
            info["total_ram_gb"] = "Unknown"
            info["available_ram_gb"] = "Unknown"

        # PyTorch & CUDA status
        try:
            import torch
            info["torch_version"] = torch.__version__
            info["cuda_available"] = torch.cuda.is_available()
            if torch.cuda.is_available():
                info["gpu_device_name"] = torch.cuda.get_device_name(0)
                info["cuda_version"] = torch.version.cuda
                info["gpu_vram_total_gb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
            else:
                info["gpu_device_name"] = "N/A (PyTorch running on CPU)"
                info["cuda_version"] = "N/A"
                info["gpu_vram_total_gb"] = 0.0
        except Exception as e:
            info["torch_version"] = "N/A"
            info["cuda_available"] = False
            info["gpu_device_name"] = f"Unavailable: {e}"

        return info

    @staticmethod
    def check_ollama_status(url: str = "http://localhost:11434") -> Dict[str, Any]:
        """Check status of Ollama without throwing exceptions."""
        status: Dict[str, Any] = {
            "ollama_url": url,
            "is_reachable": False,
            "models": [],
            "error": None
        }
        try:
            req = urllib.request.Request(f"{url}/api/tags", headers={"User-Agent": "StudyAstra-Benchmark"})
            with urllib.request.urlopen(req, timeout=2.5) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    models = [m.get("name") for m in data.get("models", [])]
                    status["is_reachable"] = True
                    status["models"] = models
        except Exception as e:
            status["error"] = str(e)

        return status


@dataclass
class BenchmarkMetric:
    """Represents an individual measured or calculated benchmark result."""
    name: str
    category: str  # "pdf_ingestion", "chunking", "embedding", "vector_db", "ollama", "conversational_rag", "startup"
    status: str    # "measured", "derived", "estimated", "unavailable"
    repetition_count: int = 1
    scale: Optional[str] = None
    cold_latency_ms: Optional[float] = None
    warm_latency_ms: Optional[float] = None
    avg_latency_ms: Optional[float] = None
    min_latency_ms: Optional[float] = None
    max_latency_ms: Optional[float] = None
    throughput_value: Optional[float] = None
    throughput_unit: Optional[str] = None
    heap_peak_mb: Optional[float] = None
    process_rss_mb: Optional[float] = None
    gpu_vram_mb: Optional[float] = None
    notes: str = ""
    limitations: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
