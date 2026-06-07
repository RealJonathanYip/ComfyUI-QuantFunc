"""Filesystem helper for building staging trees that point at source weights.

A staging dir references the original (multi-GB) weight files instead of copying
them. The cheapest reference is an OS link; this helper picks the best one
available and degrades gracefully:

    hardlink  → symlink → copy

Why this order:
  * hardlink (os.link): zero extra disk (shares the inode), instant, and —
    crucially — needs NO special privilege on Windows (unlike symlinks). Its
    only constraint is that source and destination live on the SAME volume.
    This is the preferred path; callers stage onto the source's volume so it
    succeeds (see factory._choose_staging_root).
  * symlink: zero-copy and cross-volume, but on Windows os.symlink raises
    WinError 1314 ("a required privilege is not held") unless Developer Mode or
    admin is on. Fine on Linux/macOS (unprivileged, cross-filesystem).
  * copy: last resort when the destination is on a different volume than the
    source AND symlinks aren't permitted (e.g. Windows, no Developer Mode,
    model on D: but staging forced onto C:). Correct but slower and uses disk.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

# Warn at most once per process when we fall back to a full copy, so the user
# learns why staging is slow without spamming the log for every shard.
_warned_copy = False


def link_or_copy(source_path: str | os.PathLike, dst: str | os.PathLike) -> str:
    """Materialise ``dst`` so it resolves to ``source_path``'s bytes.

    Tries hardlink, then symlink, then copy (see module docstring). ``dst`` is
    replaced if it already exists; its parent directory must already exist.
    Returns the method used: ``"hardlink"`` | ``"symlink"`` | ``"copy"``.
    """
    global _warned_copy
    src = os.path.abspath(os.fspath(source_path))
    dst = Path(dst)
    if dst.is_symlink() or dst.exists():
        dst.unlink()

    # 1. hardlink — same volume, no privilege, no extra bytes.
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError as e_hard:
        # 2. symlink — cross-volume, zero-copy; needs privilege on Windows.
        try:
            dst.symlink_to(src)
            return "symlink"
        except OSError as e_sym:
            # 3. copy — always works.
            if not _warned_copy:
                logger.warning(
                    "[staging] cannot hardlink or symlink weights into the "
                    "staging dir (hardlink: %s; symlink: %s) — copying instead. "
                    "This is slower and uses extra disk. On Windows this happens "
                    "when staging lands on a different drive than the model; set "
                    "QUANTFUNC_CACHE_DIR to a folder on the model's drive (or "
                    "enable Developer Mode) to use instant links.",
                    e_hard, e_sym)
                _warned_copy = True
            shutil.copy2(src, dst)
            return "copy"
