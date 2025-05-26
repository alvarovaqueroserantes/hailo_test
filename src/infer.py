import numpy as np
import logging
import time
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
        logging.info(f"Running inference on input shape: {input_tensor.shape}")
        time.sleep(0.2)  # Simulated latency
        output = np.random.rand(10)
        output /= np.sum(output)
        return [output]

# ----------------------------
# Utilities
# ----------------------------
def generate_fake_image():
    """
    Simulates one RGB image (1 x 3 x 224 x 224)
    """
    return np.random.rand(1, 3, 224, 224).astype(np.float32)

def log_prediction(pred_probs, iteration):
    top_indices = np.argsort(pred_probs)[::-1][:3]
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    print(f"\n🕒 [{timestamp}] Iteration {iteration}")
    print("Top-3 Predictions:")
    for i, idx in enumerate(top_indices, start=1):
        conf = pred_probs[idx] * 100
        print(f"  #{i} - Class {idx} : {conf:.2f}%")

    top_class = top_indices[0]
    confidence = pred_probs[top_class]
    if confidence > 0.6:
        print(f"Final Prediction: CLASS {top_class} with HIGH confidence ({confidence*100:.2f}%)")
    else:
        print(f"Final Prediction: CLASS {top_class} with LOW confidence ({confidence*100:.2f}%)")

# ----------------------------
# Main Execution Loop
# ----------------------------
def main():
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%H:%M:%S'
    )

    hef_path = "models/fake_model.hef"
    num_iterations =  10000 # Simulate N streaming frames

    logging.info("Starting continuous inference pipeline...")

    with FakeHef(hef_path) as hef:
        with FakeInferer(hef) as inferer:
            for i in range(1, num_iterations + 1):
                input_tensor = generate_fake_image()
                start_time = time.time()

                output_probs = inferer.infer(input_tensor)[0]

                elapsed = (time.time() - start_time) * 1000
                logging.info(f"Inference completed in {elapsed:.2f} ms")
                
                log_prediction(output_probs, i)

    logging.info("✅ Finished simulation.")

if __name__ == "__main__":
    main()
