import pytest
import sys
import os
import numpy as np
import cv2

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.storage import MinioClient

@pytest.fixture(scope="session")
def minio_client():
    return MinioClient()

@pytest.fixture
def dummy_frame():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(frame, (100, 100), (200, 200), (0, 255, 0), -1)
    return frame

@pytest.fixture
def dummy_bbox():
    # [x1, y1, x2, y2]
    return [100, 100, 200, 200]
