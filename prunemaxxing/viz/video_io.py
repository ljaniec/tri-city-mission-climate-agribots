"""Shared H.264 writer for every showcase video."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import imageio.v2 as imageio


@contextmanager
def video_writer(out_path: Path, fps: int, crf: int = 20) -> Iterator:
    """Encode to a hidden temp file and swap it in on success.

    An MP4 is unplayable until its index (moov atom) is written on close, so the
    previous complete video stays in place while a re-render is running.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(f".{out_path.stem}.partial{out_path.suffix}")
    writer = imageio.get_writer(tmp, fps=fps, codec="libx264", quality=None, macro_block_size=8,
                                pixelformat="yuv420p", ffmpeg_params=["-crf", str(crf), "-preset", "medium"])
    try:
        yield writer
    except BaseException:
        writer.close()
        tmp.unlink(missing_ok=True)
        raise
    writer.close()
    os.replace(tmp, out_path)
