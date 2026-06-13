import cv2
import numpy as np
import argparse
import os
import time
from pathlib import Path
# ─── Konfigurasi 
CONFIG = {
    "weights": "yolov4_birds.weights",   
    "config":  "yolov4_birds.cfg",       
    "classes": "bird_classes.txt",       
    "conf_threshold": 0.5,               
    "nms_threshold":  0.4,               
    "input_size":     (416, 416),        
}
# Warna per kelas (BGR)
COLORS = [
    (0,   165, 255),   # Burung Gereja   - Orange
    (0,   255,   0),   # Burung Pipit    - Hijau
    (255,  50,  50),   # Burung Bondol   - Merah
    (255, 255,   0),   # Burung Emprit   - Kuning
    (200,   0, 255),   # Burung Lainnya  - Ungu
]
# ─── Kelas-kelas Burung Hama Padi 
BIRD_CLASSES_DEFAULT = [
    "Burung Gereja (Passer montanus)",
    "Burung Pipit (Lonchura leucogastroides)",
    "Burung Bondol Peking (Lonchura punctulata)",
    "Burung Emprit (Lonchura striata)",
    "Burung Hama Lainnya",
]

def load_classes(filepath: str) -> list:
    """Muat daftar kelas dari file teks."""
    if os.path.exists(filepath):
        with open(filepath) as f:
            return [line.strip() for line in f if line.strip()]
    print(f"[PERINGATAN] File kelas '{filepath}' tidak ditemukan. Menggunakan default.")
    return BIRD_CLASSES_DEFAULT
def load_yolo_model(cfg_path: str, weights_path: str):
    """
    Muat model YOLOv4 menggunakan OpenCV DNN.
    Jika file tidak ada, kembalikan None (mode demo/simulasi).
    """
    if not os.path.exists(cfg_path) or not os.path.exists(weights_path):
        print("[INFO] File model tidak ditemukan – berjalan dalam MODE SIMULASI.")
        return None

    net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)

    # Gunakan GPU jika tersedia
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return net, output_layers

def detect_birds(net, output_layers, frame: np.ndarray,
                 classes: list, conf_thr: float, nms_thr: float,
                 input_size: tuple) -> list:
    """
    Jalankan inferensi YOLOv4 pada satu frame.
    Mengembalikan list deteksi: [(label, confidence, x, y, w, h), ...]
    """
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        frame, 1/255.0, input_size,
        swapRB=True, crop=False
    )
    net.setInput(blob)
    layer_outputs = net.forward(output_layers)
    boxes, confidences, class_ids = [], [], []
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = int(np.argmax(scores))
            confidence = float(scores[class_id])
            if confidence < conf_thr:
                continue
            cx, cy, bw, bh = detection[:4]
            x = int((cx - bw / 2) * w)
            y = int((cy - bh / 2) * h)
            boxes.append([x, y, int(bw * w), int(bh * h)])
            confidences.append(confidence)
            class_ids.append(class_id)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_thr, nms_thr)
    results = []
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, bw, bh = boxes[i]
            label = classes[class_ids[i]] if class_ids[i] < len(classes) else "Unknown"
            results.append((label, confidences[i], x, y, bw, bh))
    return results
def simulate_detections(frame: np.ndarray, classes: list) -> list:
    """Mode simulasi — menghasilkan deteksi acak untuk demo UI."""
    h, w = frame.shape[:2]
    np.random.seed(int(time.time()) % 100)
    n = np.random.randint(0, 4)
    results = []
    for _ in range(n):
        label = np.random.choice(classes[:3])
        conf  = float(np.random.uniform(0.55, 0.98))
        bw    = np.random.randint(w // 10, w // 4)
        bh    = np.random.randint(h // 10, h // 4)
        x     = np.random.randint(0, w - bw)
        y     = np.random.randint(0, h - bh)
        results.append((label, conf, x, y, bw, bh))
    return results
def draw_detections(frame: np.ndarray, detections: list, classes: list) -> np.ndarray:
    """Gambar bounding box dan label pada frame."""
    overlay = frame.copy()
    for (label, conf, x, y, bw, bh) in detections:
        idx   = classes.index(label) if label in classes else 0
        color = COLORS[idx % len(COLORS)]
        # Bounding box
        cv2.rectangle(overlay, (x, y), (x + bw, y + bh), color, 2)
        # Background label
        text  = f"{label.split('(')[0].strip()} {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(overlay, (x, y - th - 8), (x + tw + 6, y), color, -1)
        cv2.putText(overlay, text, (x + 3, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    # Blend untuk efek semi-transparan
    result = cv2.addWeighted(overlay, 0.85, frame, 0.15, 0)
    # ── Overlay statistik ──
    total = len(detections)
    status = "⚠ HAMA TERDETEKSI" if total > 0 else "✓ AMAN"
    color_status = (0, 60, 255) if total > 0 else (0, 200, 80)
    cv2.rectangle(result, (10, 10), (320, 70), (0, 0, 0), -1)
    cv2.putText(result, status, (18, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_status, 2)
    cv2.putText(result, f"Jumlah burung: {total}", (18, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
    return result
def process_image(image_path: str, model_info, classes: list, cfg: dict):
    """Proses satu gambar dan simpan hasilnya."""
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[ERROR] Tidak dapat membaca gambar: {image_path}")
        return
    if model_info:
        net, output_layers = model_info
        detections = detect_birds(net, output_layers, frame, classes,
                                  cfg["conf_threshold"], cfg["nms_threshold"],
                                  cfg["input_size"])
    else:
        detections = simulate_detections(frame, classes)
    result = draw_detections(frame, detections, classes)
    out_path = Path(image_path).stem + "_detected.jpg"
    cv2.imwrite(out_path, result)
    print(f"[HASIL] {len(detections)} burung terdeteksi → disimpan ke '{out_path}'")
    for d in detections:
        print(f"  • {d[0]}  conf={d[1]:.2f}  box=({d[2]},{d[3]},{d[4]},{d[5]})")

    cv2.imshow("Deteksi Hama Burung Padi – YOLOv4", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def process_video(source, model_info, classes: list, cfg: dict):
    """Proses video / webcam secara real-time."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Tidak dapat membuka sumber video: {source}")
        return
    fps_target = 30
    prev_time  = time.time()
    print("[INFO] Tekan 'q' untuk keluar, 's' untuk screenshot.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if model_info:
            net, output_layers = model_info
            detections = detect_birds(net, output_layers, frame, classes,
                                      cfg["conf_threshold"], cfg["nms_threshold"],
                                      cfg["input_size"])
        else:
            detections = simulate_detections(frame, classes)
        result = draw_detections(frame, detections, classes)
        # Tampilkan FPS
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time + 1e-6)
        prev_time = curr_time
        cv2.putText(result, f"FPS: {fps:.1f}", (result.shape[1] - 110, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

        cv2.imshow("Deteksi Hama Burung Padi – YOLOv4", result)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            fname = f"screenshot_{int(time.time())}.jpg"
            cv2.imwrite(fname, result)
            print(f"[INFO] Screenshot disimpan → {fname}")
    cap.release()
    cv2.destroyAllWindows()
def main():
    parser = argparse.ArgumentParser(
        description="Deteksi Hama Burung Padi – YOLOv4")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--input",  help="Path gambar atau video")
    group.add_argument("--webcam", action="store_true",
                       help="Gunakan webcam (kamera 0)")
    parser.add_argument("--weights", default=CONFIG["weights"])
    parser.add_argument("--config",  default=CONFIG["config"])
    parser.add_argument("--classes", default=CONFIG["classes"])
    parser.add_argument("--conf",    type=float, default=CONFIG["conf_threshold"])
    parser.add_argument("--nms",     type=float, default=CONFIG["nms_threshold"])
    args = parser.parse_args()
    cfg = {**CONFIG, "conf_threshold": args.conf, "nms_threshold": args.nms}
    classes    = load_classes(args.classes)
    model_info = load_yolo_model(args.config, args.weights)
    if args.webcam:
        process_video(0, model_info, classes, cfg)
    elif args.input:
        ext = Path(args.input).suffix.lower()
        if ext in (".jpg", ".jpeg", ".png", ".bmp"):
            process_image(args.input, model_info, classes, cfg)
        elif ext in (".mp4", ".avi", ".mov", ".mkv"):
            process_video(args.input, model_info, classes, cfg)
        else:
            print("[ERROR] Format file tidak dikenali.")
    else:
        print("[INFO] Tidak ada input – membuka webcam default.")
        process_video(0, model_info, classes, cfg)
if __name__ == "__main__":
    main()

