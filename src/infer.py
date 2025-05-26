import numpy as np
import logging
from datetime import datetime

# ----------------------------
# Simulated Hailo SDK Classes
# ----------------------------
class FakeHef:
    def __init__(self, path):
        logging.info(f"Loaded fake HEF model from: {path}")
    def __enter__(self): return self
    def __exit__(self, exc_type, exc_val, exc_tb): pass

class FakeInferer:
    def __init__(self, hef):
        logging.info("Initialized fake Hailo inferer.")
    def __enter__(self): return self
    def __exit__(self, exc_type, exc_val, exc_tb): pass
    def infer(self, input_tensor):
        logging.info(f"Running inference on input with shape: {input_tensor.shape}")
        # Simulate a classification output (10 classes)
        fake_output = np.random.rand(10)
        fake_output /= np.sum(fake_output)  # Normalize to probabilities
        return [fake_output]

# ----------------------------
# Utilities
# ----------------------------
def generate_fake_image(batch_size=1):
    """
    Simulate a batch of RGB images (batch_size x 3 x 224 x 224)
    """
    return np.random.rand(batch_size, 3, 224, 224).astype(np.float32)

def log_prediction(pred_probs):
    """
    Log top prediction and top-3 scores.
    """
    top_indices = np.argsort(pred_probs)[::-1][:3]
    logging.info("Top Predictions:")
    for rank, idx in enumerate(top_indices, start=1):
        logging.info(f"#{rank}: Class {idx} — Confidence: {pred_probs[idx]:.4f}")

    top_prediction = top_indices[0]
    confidence = pred_probs[top_prediction]
    print(f"\n🔮 Final Prediction: CLASS {top_prediction} with {confidence*100:.2f}% confidence")

# ----------------------------
# Main Execution
# ----------------------------
def main():
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%H:%M:%S'
    )

    logging.info("Starting fake Hailo AI inference pipeline...")

    hef_path = "models/fake_model.hef"
    input_tensor = generate_fake_image()

    with FakeHef(hef_path) as hef:
        with FakeInferer(hef) as inferer:
            output_probs = inferer.infer(input_tensor)[0]
            log_prediction(output_probs)

    logging.info("Inference complete.")

if __name__ == "__main__":
    main()
