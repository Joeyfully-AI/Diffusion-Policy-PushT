from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import imageio.v2 as imageio


@dataclass
class VideoRecorder:
    fps: int = 10
    codec: str = 'h264'
    input_pix_fmt: str = 'rgb24'
    crf: int = 22
    thread_type: str = 'FRAME'
    thread_count: int = 1

    def __post_init__(self):
        self._writer = None
        self._file_path: Optional[str] = None

    @classmethod
    def create_h264(
        cls,
        fps=10,
        codec='h264',
        input_pix_fmt='rgb24',
        crf=22,
        thread_type='FRAME',
        thread_count=1,
    ) -> 'VideoRecorder':
        return cls(
            fps=fps,
            codec=codec,
            input_pix_fmt=input_pix_fmt,
            crf=crf,
            thread_type=thread_type,
            thread_count=thread_count,
        )

    def is_ready(self) -> bool:
        return self._writer is not None

    def start(self, file_path: str) -> None:
        if self._writer is not None:
            self.stop()
        self._file_path = file_path
        self._writer = imageio.get_writer(
            file_path,
            fps=self.fps,
            codec=self.codec,
            format='FFMPEG',
            macro_block_size=None,
        )

    def write_frame(self, frame) -> None:
        if self._writer is None:
            raise RuntimeError('VideoRecorder is not started')
        self._writer.append_data(frame)

    def stop(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None
            self._file_path = None

    def __del__(self):
        try:
            self.stop()
        except Exception:
            pass
