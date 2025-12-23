import pytest
from core.vehicle import Vehicle
import numpy as np
from unittest.mock import MagicMock

def test_vehicle_initialization(dummy_bbox):
    vehicle = Vehicle(dummy_bbox, class_id=1)
    assert vehicle.class_id == 1
    assert vehicle.has_violated is False
    assert len(vehicle.violation_type) == 0

def test_vehicle_mark_violation(dummy_bbox, dummy_frame):
    vehicle = Vehicle(dummy_bbox, class_id=1)
    vehicle.id = 123 # Mock ID
    
    # Pre-set license plate votes (simulating continuous detection before mark_violation)
    vehicle.lp_votes = {"ABC-123": 3}
    vehicle.license_plate = "ABC-123"
    
    vehicle.has_violated = True
    
    # Mark violation - no recognizer needed (LP detection is now handled by ViolationManager)
    dummy_buffer = [(i, dummy_frame) for i in range(5)]
    state = dummy_bbox  # [x1, y1, x2, y2]
    vehicle.mark_violation("RedLight", frame=dummy_frame, frame_buffer=dummy_buffer, fps=10, state=state)
    
    assert vehicle.has_violated is None
    assert "RedLight" in vehicle.violation_type
    assert len(vehicle.proof) > 0
    assert vehicle.proof.shape[2] == 3 # Check channels
    
    # Check if proof is a crop (smaller or equal to frame)
    assert vehicle.proof.size > 0

def test_vehicle_mark_violation_no_frame(dummy_bbox):
    """Test marking violation without a frame (should still update state but no proof)"""
    vehicle = Vehicle(dummy_bbox, class_id=1)
    
    # Pre-set license plate votes
    vehicle.lp_votes = {"ABC-123": 1}
    
    vehicle.has_violated = True
    vehicle.mark_violation("Speeding")
    
    assert "Speeding" in vehicle.violation_type
    assert len(vehicle.proof) == 0

def test_vehicle_lp_voting(dummy_bbox):
    """Test license plate voting logic"""
    vehicle = Vehicle(dummy_bbox, class_id=1)
    
    # Add votes
    vehicle.update_license_plate("ABC-123")
    vehicle.update_license_plate("ABC-123")
    assert vehicle.license_plate is None  # Below threshold (3)
    
    vehicle.update_license_plate("XYZ-999")
    vehicle.update_license_plate("ABC-123")
    assert vehicle.license_plate == "ABC-123"  # Threshold met
