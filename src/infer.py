import numpy as np
import logging
import time
from datetime import datetime
from hailo_platform import hailort

# ----------------------------
# Utilities
# ----------------------------
def generate_fake_image():
    """
    Simula una imagen RGB normalizada de 224x224 (1 x 3 x 224 x 224)
    """
    img = np.random.rand(1, 3, 224, 224).astype(np.float32)
    return img

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
        print(f"✅ Final Prediction: CLASS {top_class} with HIGH confidence ({confidence*100:.2f}%)")
    else:
        print(f"⚠️ Final Prediction: CLASS {top_class} with LOW confidence ({confidence*100:.2f}%)")

# ----------------------------
# Main with Hailo Runtime
# ----------------------------
def main():
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%H:%M:%S'
    )

    hef_path = "models/mobilenet_v1.hef"
    num_iterations = 100

    logging.info("🚀 Starting Hailo-8 inference pipeline...")

    with hailort.Hef(hef_path) as hef:
        network_groups = hef.get_network_groups()
        with hailort.Inferer(network_groups[0]) as inferer:
            for i in range(1, num_iterations + 1):
                input_tensor = generate_fake_image()

                start_time = time.time()
                results = inferer.infer(input_tensor)[0]
                elapsed = (time.time() - start_time) * 1000

                logging.info(f"Inference completed in {elapsed:.2f} ms")
                log_prediction(results, i)

    logging.info("✅ Finished Hailo-8 inference.")

if __name__ == "__main__":
    main()
