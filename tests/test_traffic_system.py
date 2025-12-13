import pytest
import cv2
from unittest.mock import MagicMock, patch
import numpy as np
from core.traffic_system import TrafficSystem

@pytest.fixture
def mock_config():
    return {
        'system': {
            'data_path': 'test_video.mp4',
            'tracker': 'bytetrack',
            'device': 'cpu',
            'vehicle_model': 'v_model.pt',
            'license_model': 'l_model.pt'
        },
        'detections': {
            'classes': [0],
            'imgsz': 640,
            'iou_threshold': 0.5,
             'conf_threshold': 0.25
        },
        'tracking': {
            'bytetrack': {
                'cost_function': 'iou',
                'max_age': 1,
                'min_hits': 1,
                'high_conf_threshold': 0.5,
                'low_conf_threshold': 0.1,
                'high_conf_iou_threshold': 0.5,
                'low_conf_iou_threshold': 0.4,
                'conf_threshold': 0.1
            }
        },
        'violation': {
            'fps': 30,
            'video_proof_duration': 1,
            'padding': 10
        }
    }

@patch('core.traffic_system.load_config')
@patch('core.traffic_system.YOLO')
@patch('core.traffic_system.PaddleOCR')
@patch('core.traffic_system.violation_save_worker')
def test_traffic_system_initialization(mock_worker, mock_ocr, mock_yolo, mock_load_config, mock_config):
    mock_load_config.return_value = mock_config
    
    system = TrafficSystem("dummy_config.yaml")
    
    # Check if models loaded with correct paths
    # Note: call_args_list might be empty if we mocked YOLO class itself differently,
    # but here we mocked the class constructor.
    assert system.vehicle_model_path == 'v_model.pt'
    assert system.license_model_path == 'l_model.pt'
    assert system.device == 'cpu'
    assert system.data_path == 'test_video.mp4'

@patch('core.traffic_system.load_config')
@patch('core.traffic_system.YOLO')
@patch('core.traffic_system.PaddleOCR')
@patch('core.traffic_system.violation_save_worker')
def test_update_config(mock_worker, mock_ocr, mock_yolo, mock_load_config, mock_config):
    mock_load_config.return_value = mock_config
    system = TrafficSystem()
    
    new_config = mock_config.copy()
    new_config['violation']['fps'] = 60
    
    system.update_config(new_config)
    assert system.config['violation']['fps'] == 60

@patch('core.traffic_system.load_config')
@patch('core.traffic_system.YOLO')
@patch('core.traffic_system.PaddleOCR')
@patch('core.traffic_system.violation_save_worker')
def test_capture_first_frame(mock_worker, mock_ocr, mock_yolo, mock_load_config, mock_config):
    mock_load_config.return_value = mock_config
    system = TrafficSystem()
    
    # Case 1: No frame
    assert system.capture_first_frame() is None
    
    # Case 2: Frame exists
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    system.first_frame = dummy_frame
    
    captured = system.capture_first_frame()
    assert captured is not None
    assert captured.shape == (100, 100, 3)

@patch('core.traffic_system.load_config')
@patch('core.traffic_system.YOLO')
@patch('core.traffic_system.PaddleOCR')
@patch('core.traffic_system.violation_save_worker')
@patch('core.traffic_system.inference_video')
def test_process_flow(mock_inference, mock_worker, mock_ocr, mock_yolo, mock_load_config, mock_config):
    mock_load_config.return_value = mock_config
    system = TrafficSystem()
    
    # Mock inference to return one result
    mock_result = MagicMock()
    mock_result.orig_img = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_inference.return_value = [mock_result]
    
    # Mock tracker instance
    system.tracker_instance = MagicMock()
    system.tracker_instance.update.return_value = [] # No tracked objects
    
    # Mock preprocess
    with patch('core.traffic_system.preprocess_detection_result') as mock_preprocess:
        mock_preprocess.return_value = (mock_result.orig_img, MagicMock())
        
        # Mock zones
        with patch('core.traffic_system.load_zones') as mock_load_zones:
            mock_load_zones.return_value = {
                'polygon': [[0, 0], [100, 0], [100, 100], [0, 100]],
                'lines_config': {
                    'violation_lines': [[0, 50], [100, 50]]
                }
            }
            
            # Start generator
            system.running = True
            generator = system._process_flow()
            
            # Get first yield
            try:
                frame, stats = next(generator)
                assert frame is not None
                assert isinstance(frame, np.ndarray)
            except StopIteration:
                pytest.fail("Generator stopped unexpectedly")

