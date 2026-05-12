"""QThread 브릿지. UI ↔ module/ 사이의 유일한 백그라운드 경로."""

from PyQt5.QtCore import QThread, pyqtSignal

from module.parser import parse_alms_bin
import module.stft as _stft


class ParseWorker(QThread):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, filepath: str):
        super().__init__()
        self.filepath = filepath

    def run(self):
        try:
            data = parse_alms_bin(self.filepath)
            self.finished.emit(data)
        except Exception as e:
            self.error.emit(str(e))


class STFTWorker(QThread):
    """perform_stft_all_ch 를 백그라운드 스레드에서 실행."""
    finished = pyqtSignal(list)
    error = pyqtSignal(str)
    progress = pyqtSignal(int)

    def __init__(self, alms_data, output_dir: str):
        super().__init__()
        self.alms_data = alms_data
        self.output_dir = output_dir

    def run(self):
        try:
            results = _stft.perform_stft_all_ch(
                output_dir=self.output_dir,
                alms_data=self.alms_data,
            )
            self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))
