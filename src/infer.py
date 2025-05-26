import numpy as np

# Simulate Hailo SDK classes and behavior
class FakeHef:
    def __init__(self, path):
        print(f"Loaded fake HEF model from: {path}")
    def __enter__(self): return self
    def __exit__(self, exc_type, exc_val, exc_tb): pass

class FakeInferer:
    def __init__(self, hef):
        print("Initialized fake Hailo inferer.")
    def __enter__(self): return self
    def __exit__(self, exc_type, exc_val, exc_tb): pass
    def infer(self, input_tensor):
        print(f"Running inference on input shape: {input_tensor.shape}")
        # Fake prediction: pretend it's a classification with 10 classes
        fake_output = np.random.rand(10)
        return [fake_output]

def generate_fake_image():
    # Simulate a 224x224 RGB image, normalized
    img = np.random.rand(3, 224, 224).astype(np.float32)
    img = np.expand_dims(img, axis=0)  # Shape: (1, 3, 224, 224)
    return img

def main():
    hef_path = "models/fake_model.hef"  # Not a real file

    input_data = generate_fake_image()

    with FakeHef(hef_path) as hef:
        with FakeInferer(hef) as inferer:
            result = inferer.infer(input_data)[0]
            predicted = np.argmax(result)
            print(f"[FAKE] Predicted class index: {predicted}")

if __name__ == "__main__":
    main()
