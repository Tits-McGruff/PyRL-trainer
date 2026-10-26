"""Best-effort scheduling for isolated trainer CLIs; importing a library changes nothing."""

import ctypes
from ctypes import wintypes
import json
import os


def _windows_api():
    """Bind only current-process priority calls with explicit pointer-sized handles."""
    kernel = getattr(ctypes, "WinDLL")("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.argtypes, kernel.GetCurrentProcess.restype = [], wintypes.HANDLE
    kernel.GetPriorityClass.argtypes = [wintypes.HANDLE]
    kernel.GetPriorityClass.restype = wintypes.DWORD
    kernel.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.SetPriorityClass.restype = wintypes.BOOL
    return kernel


def current_process_priority():
    """Read this process only; priority is diagnostic policy, not recurrent or optimizer state."""
    if os.name == "nt":
        kernel = _windows_api()
        value = kernel.GetPriorityClass(kernel.GetCurrentProcess())
        if not value:
            raise OSError(getattr(ctypes, "get_last_error")(),
                          "Windows trainer process priority call failed")
        return {"platform": "windows", "priorityClass": int(value)}
    value = getattr(os, "getpriority")(getattr(os, "PRIO_PROCESS"), 0)
    return {"platform": "posix", "nice": value}


def configure_cli_priority():
    """Lower this standalone trainer before Torch imports; never raise an existing low priority.

    Windows: https://learn.microsoft.com/windows/win32/procthread/scheduling-priorities
    POSIX: https://docs.python.org/3/library/os.html#os.setpriority
    """
    before = current_process_priority()
    if os.name == "nt":
        # Below Normal (0x4000) and Idle (0x40) already yield to normal world authority.
        if before["priorityClass"] not in (0x4000, 0x40):
            kernel = _windows_api()
            if not kernel.SetPriorityClass(kernel.GetCurrentProcess(), 0x4000):
                raise OSError(getattr(ctypes, "get_last_error")(),
                          "Windows trainer process priority call failed")
    else:
        getattr(os, "setpriority")(getattr(os, "PRIO_PROCESS"), 0, max(5, before["nice"]))
    return current_process_priority()


if __name__ == "__main__":
    print(json.dumps(configure_cli_priority(), sort_keys=True))
