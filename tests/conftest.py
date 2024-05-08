import cv2
import pytest

FEATURES = 256


# create an image for testing from a local image
@pytest.fixture(scope="session")
def image():
    path = "tests/data/lena.png"
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError(f"Image not found or unable to load: {path}")

    return image
