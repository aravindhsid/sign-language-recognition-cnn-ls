"""
Dataset Recorder Utility for Sign Language Gestures.
Interactive webcam tool to record real sign language video clips:
- Provides visual countdown (3, 2, 1, Action!) before recording.
- Saves clips directly to data/raw_videos/<class_name>/clip_XX.mp4.
- Enables users to build customized real-world sign datasets.
"""

import os
import time
from pathlib import Path
import cv2
import config

RECORD_FPS = 20.0
RECORD_DURATION_SEC = 2.0  # Duration per gesture clip
FRAMES_PER_CLIP = int(RECORD_FPS * RECORD_DURATION_SEC)
TARGET_RESOLUTION = (config.FRAME_WIDTH * 2, config.FRAME_HEIGHT * 2)  # 224x224


def record_gesture_samples(
    classes=config.DEFAULT_CLASSES,
    samples_per_class=5,
    camera_id=0
):
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Error: Could not access camera with ID {camera_id}.")
        return

    print("=" * 65)
    print("      SIGN LANGUAGE GESTURE DATASET RECORDER")
    print("=" * 65)
    print("Instructions:")
    print(" - Press 'SPACE' to begin countdown for next clip")
    print(" - Press 'N' to skip to next class")
    print(" - Press 'Q' or 'ESC' to exit")
    print("=" * 65)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    for class_name in classes:
        class_dir = config.RAW_VIDEOS_DIR / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        existing_clips = list(class_dir.glob("*.mp4"))
        sample_count = len(existing_clips)

        print(f"\n>>> Current Gesture Class: [{class_name}] (Existing: {sample_count} clips)")

        while sample_count < samples_per_class:
            ready_to_record = False

            # Idle loop waiting for user trigger
            while not ready_to_record:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)  # Mirror view
                h, w, _ = frame.shape

                # Guide overlay
                cv2.rectangle(frame, (w // 4, h // 6), (3 * w // 4, 5 * h // 6), (0, 255, 0), 2)
                cv2.putText(
                    frame, f"Class: {class_name} ({sample_count + 1}/{samples_per_class})",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2
                )
                cv2.putText(
                    frame, "Press SPACE to Record | Q to Quit",
                    (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2
                )

                cv2.imshow("Sign Language Dataset Recorder", frame)
                key = cv2.waitKey(20) & 0xFF
                if key == 32:  # Spacebar
                    ready_to_record = True
                elif key in (ord('n'), ord('N')):
                    break
                elif key in (27, ord('q'), ord('Q')):
                    cap.release()
                    cv2.destroyAllWindows()
                    return

            if not ready_to_record:
                break  # Skipped to next class

            # Countdown 3, 2, 1
            for sec in [3, 2, 1]:
                start_t = time.time()
                while time.time() - start_t < 1.0:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = cv2.flip(frame, 1)
                    h, w, _ = frame.shape
                    cv2.putText(
                        frame, str(sec), (w // 2 - 20, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 0, 255), 6
                    )
                    cv2.imshow("Sign Language Dataset Recorder", frame)
                    cv2.waitKey(10)

            # Record Clip
            sample_count += 1
            output_file = class_dir / f"clip_{sample_count:03d}.mp4"
            out = cv2.VideoWriter(str(output_file), fourcc, RECORD_FPS, TARGET_RESOLUTION)

            recorded_frames = 0
            while recorded_frames < FRAMES_PER_CLIP:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                resized = cv2.resize(frame, TARGET_RESOLUTION)
                out.write(resized)

                # Recording indicator
                cv2.circle(frame, (30, 30), 12, (0, 0, 255), -1)
                cv2.putText(frame, "RECORDING", (50, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow("Sign Language Dataset Recorder", frame)
                cv2.waitKey(int(1000 / RECORD_FPS))
                recorded_frames += 1

            out.release()
            print(f"  -> Saved clip: {output_file.name}")

    cap.release()
    cv2.destroyAllWindows()
    print("\nDataset recording completed successfully!")


if __name__ == "__main__":
    record_gesture_samples(samples_per_class=5)
