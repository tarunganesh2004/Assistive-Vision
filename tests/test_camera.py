import pytest
from src.camera import Camera


def test_camera_init():
    cam = Camera(source=0)
    assert cam.cap is None
    cam.start()
    assert cam.cap.isOpened()
    cam.release()
