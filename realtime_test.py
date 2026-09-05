# realtime_eval.py
import cv2, time, csv, os, math
import numpy as np
from tensorflow.keras.models import load_model
from cvzone.HandTrackingModule import HandDetector

# ------------ config ------------
MODEL_PATH = "mudra_cnn_model.keras"      # your trained model
LABELS = ['Musti','Pataka','Sikhara','Simhamukha','Trisula']
IMG_SIZE = 300
OFFSET = 20
MAXHANDS = 1

TRIAL_FRAMES = 60          # frames per trial (≈ 2 sec at 30 fps)
WARMUP_FRAMES = 15         # ignore first few frames in each trial
CONFIDENCE_DECISION = "majority"  # 'majority' or 'threshold'
MAJORITY_REQUIRED = 0.5    # >50% of frames must agree
CONF_THRESHOLD = 0.60      # used if CONFIDENCE_DECISION == 'threshold'

OUT_DIR = "realtime_logs"
os.makedirs(OUT_DIR, exist_ok=True)
RUN_CSV = os.path.join(OUT_DIR, f"realtime_log_{int(time.time())}.csv")

# ------------ helpers ------------
def preprocess_from_bbox(frame, bbox):
    x,y,w,h = bbox
    imgWhite = np.ones((IMG_SIZE, IMG_SIZE, 3), np.uint8)*255
    imgCrop = frame[max(0,y - OFFSET):y + h + OFFSET, max(0,x - OFFSET):x + w + OFFSET]

    if imgCrop.size == 0:
        # fallback if bbox spills out of frame
        return None

    aspect = h / float(w + 1e-6)
    if aspect > 1:
        k = IMG_SIZE / float(h)
        wCal = math.ceil(k * w)
        imgResize = cv2.resize(imgCrop, (wCal, IMG_SIZE))
        wGap = (IMG_SIZE - wCal) // 2
        imgWhite[:, wGap:wGap + wCal] = imgResize
    else:
        k = IMG_SIZE / float(w)
        hCal = math.ceil(k * h)
        imgResize = cv2.resize(imgCrop, (IMG_SIZE, hCal))
        hGap = (IMG_SIZE - hCal) // 2
        imgWhite[hGap:hGap + hCal, :] = imgResize

    imgInput = imgWhite.astype("float32") / 255.0
    return np.expand_dims(imgInput, 0)  # (1, 300, 300, 3)

# ------------ main ------------
def main():
    model = load_model(MODEL_PATH)
    cap = cv2.VideoCapture(0)
    detector = HandDetector(maxHands=MAXHANDS)
    print(f"✔ Model loaded. Logging to: {RUN_CSV}")

    with open(RUN_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["trial_id","target_label","frame_idx","pred_label","pred_conf","latency_ms","ts"])

        trial_id = 0
        while True:
            # cycle through mudras; you can press 'n' to move to next, 'r' to repeat, 'q' to quit
            target_label = LABELS[trial_id % len(LABELS)]
            print(f"\n=== Trial {trial_id} — Show: {target_label} ===")
            print("Hold the mudra steady. Recording will start when you press 's'. Press 'q' to exit.")
            # prompt
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Camera read failed.")
                    return
                disp = frame.copy()
                cv2.putText(disp, f"Ready: press 's' to start | target: {target_label}", (20,40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                cv2.imshow("Realtime Eval", disp)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    cap.release(); cv2.destroyAllWindows(); return
                if key == ord('s'):
                    break

            preds = []      # (label_idx, conf)
            latencies = []
            start_ts = time.time()

            for fi in range(TRIAL_FRAMES):
                t0 = time.time()
                ret, frame = cap.read()
                if not ret:
                    break

                hands, img_draw = detector.findHands(frame, draw=True)
                if hands:
                    bbox = hands[0]['bbox']
                    inp = preprocess_from_bbox(frame, bbox)
                    if inp is not None:
                        pred = model.predict(inp, verbose=0)[0]
                        idx = int(np.argmax(pred))
                        conf = float(np.max(pred))
                        preds.append((idx, conf))
                        latency_ms = (time.time() - t0)*1000.0
                        latencies.append(latency_ms)

                        label_text = f"Pred: {LABELS[idx]} ({conf:.2f})"
                        x,y,w,h = bbox
                        cv2.rectangle(img_draw, (x - OFFSET, y - OFFSET), (x + w + OFFSET, y + h + OFFSET), (0,255,0), 2)
                        cv2.putText(img_draw, label_text, (x, max(30, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(0,255,0),2)

                        # log every frame
                        writer.writerow([trial_id, target_label, fi, LABELS[idx], conf, latency_ms, time.time()])

                cv2.putText(img_draw, f"Trial {trial_id} | Target: {target_label}", (20,30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                cv2.putText(img_draw, f"Frame {fi+1}/{TRIAL_FRAMES}", (20,60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                cv2.imshow("Realtime Eval", img_draw)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cap.release(); cv2.destroyAllWindows(); return

            # make a trial decision
            preds_valid = preds[WARMUP_FRAMES:] if len(preds) > WARMUP_FRAMES else preds
            decided_label = "NA"
            if preds_valid:
                idxs = [p[0] for p in preds_valid]
                confs = [p[1] for p in preds_valid]

                if CONFIDENCE_DECISION == "majority":
                    # majority vote
                    counts = np.bincount(idxs, minlength=len(LABELS))
                    winner = int(np.argmax(counts))
                    frac = counts[winner] / max(1, len(idxs))
                    decided_label = LABELS[winner] if frac >= MAJORITY_REQUIRED else "Uncertain"
                else:
                    # threshold on mean confidence of winner frames
                    counts = np.bincount(idxs, minlength=len(LABELS))
                    winner = int(np.argmax(counts))
                    winner_confs = [c for (i,c) in preds_valid if i == winner]
                    decided_label = LABELS[winner] if (np.mean(winner_confs) >= CONF_THRESHOLD) else "Uncertain"

                avg_latency = np.mean(latencies) if latencies else float("nan")
                print(f"Trial {trial_id} target={target_label} -> decision={decided_label} | "
                      f"frames={len(preds_valid)} | avg_latency={avg_latency:.1f} ms")

            print("Press 'n' for next mudra, 'r' to repeat, 'q' to quit.")
            while True:
                key = cv2.waitKey(0) & 0xFF
                if key == ord('n'):
                    trial_id += 1
                    break
                elif key == ord('r'):
                    # repeat same trial id (do not increment)
                    break
                elif key == ord('q'):
                    cap.release(); cv2.destroyAllWindows(); return

if __name__ == "__main__":
    main()
