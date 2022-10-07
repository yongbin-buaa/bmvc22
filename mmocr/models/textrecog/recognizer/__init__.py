from .base import BaseRecognizer
from .crnn import CRNNNet
from .encode_decode_recognizer import EncodeDecodeRecognizer
from .nrtr import NRTR
from .robust_scanner import RobustScanner
from .sar import SARNet
from .seg_recognizer import SegRecognizer
from .fr_encode_decode_recognizer import FRNet

__all__ = [
    'BaseRecognizer', 'EncodeDecodeRecognizer', 'CRNNNet', 'SARNet', 'NRTR',
    'SegRecognizer', 'RobustScanner', 'FRNet'
]
