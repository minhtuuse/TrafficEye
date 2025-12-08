import pytest
from core.vehicle import Vehicle
import numpy as np

def test_vehicle_initialization(dummy_bbox):
    vehicle = Vehicle(dummy_bbox, class_id=1)
    assert vehicle.class_id == 1
    assert vehicle.has_violated is False
    assert len(vehicle.violation_type) == 0

def test_vehicle_mark_violation(dummy_bbox, dummy_frame):
    vehicle = Vehicle(dummy_bbox, class_id=1)
    vehicle.id = 123 # Mock ID
    
    # Mark violation
    dummy_buffer = [dummy_frame for _ in range(5)]
    vehicle.mark_violation("RedLight", frame=dummy_frame, frame_buffer=dummy_buffer, fps=10)
    
    assert vehicle.has_violated is True
    assert "RedLight" in vehicle.violation_type
    assert len(vehicle.proof) > 0
    assert vehicle.proof.shape[2] == 3 # Check channels
    
    # Check if proof is a crop (smaller or equal to frame)
    # With padding it might be slightly different logic, but definitely not empty
    assert vehicle.proof.size > 0

def test_vehicle_mark_violation_no_frame(dummy_bbox):
    """Test marking violation without a frame (should still update state but no proof)"""
    vehicle = Vehicle(dummy_bbox, class_id=1)
    vehicle.mark_violation("Speeding")
    
    assert vehicle.has_violated is True
    assert "Speeding" in vehicle.violation_type
    assert len(vehicle.proof) == 0
