"""Demo: real-time multi-person detection and tracking (YOLO + DeepSORT).

This is a minimal, self-contained demonstration of the detection/tracking
stage of the school-bullying monitoring system: YOLO detects people in
each frame, DeepSORT associates detections across frames into stable
track IDs. The behavior-recognition stage (interaction classification on
track pairs) is not included here — see the repository README.

Usage:
    python demo/track_demo.py --source path/to/video.mp4
    python demo/track_demo.py --source 0            # webcam
    python demo/track_demo.py --source video.mp4 --save out.mp4 --no-show

The first run downloads YOLO weights (~6 MB) automatically.
"""

from __future__ import annotations

import argparse

import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO

PERSON_CLASS = 0  # COCO class id for "person"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", default="0", help="video path, or webcam index")
    p.add_argument("--model", default="yolov5su.pt", help="ultralytics model weights")
    p.add_argument("--conf", type=float, default=0.35, help="detection confidence")
    p.add_argument("--save", default=None, help="write annotated video to this path")
    p.add_argument("--no-show", action="store_true", help="disable display window")
    return p.parse_args()


def main():
    args = parse_args()
    source = int(args.source) if args.source.isdigit() else args.source

    detector = YOLO(args.model)
    tracker = DeepSort(max_age=30, n_init=3)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"could not open source: {args.source}")

    writer = None
    if args.save:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(
            args.save, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
        )

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # 1) detect people
        result = detector(frame, classes=[PERSON_CLASS], conf=args.conf, verbose=False)[0]
        detections = [
            ([float(x1), float(y1), float(x2 - x1), float(y2 - y1)], float(conf), "person")
            for x1, y1, x2, y2, conf, _cls in result.boxes.data.tolist()
        ]

        # 2) associate across frames
        tracks = tracker.update_tracks(detections, frame=frame)

        # 3) draw stable track IDs
        for track in tracks:
            if not track.is_confirmed():
                continue
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
            cv2.putText(
                frame,
                f"ID {track.track_id}",
                (x1, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 200, 0),
                2,
            )

        if writer is not None:
            writer.write(frame)
        if not args.no_show:
            cv2.imshow("YOLO + DeepSORT demo", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
