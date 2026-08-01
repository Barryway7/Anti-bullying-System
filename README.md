# Intelligent Monitoring System for School-Bullying Detection

Team project at Northeastern University (01/2022–09/2023), advised by Prof. Dongyue Chen. Team leader: [Wei Fu](https://barryway7.github.io)

> **Provenance note.** The original system (trained models, curated dataset, school-camera integration) is not public — the training data includes footage that cannot be released. This repository documents the system and provides a **clearly-labeled demo** of its detection/tracking stage, written in 2026 as a minimal reproducible illustration.

## Overview

A real-time computer-vision system that detects potentially harmful physical interactions among students in crowded school environments and alerts staff so they can intervene promptly.

## System design

- **Detection & tracking.** YOLOv5 detects individuals; DeepSORT associates detections across frames into a real-time multi-target tracking pipeline that remains stable in crowded scenes.
- **Behavior recognition.** Track outputs feed an interaction classifier trained to flag violent or harmful behaviors.
- **Training data.** A public violence dataset augmented with data collected and curated by our team, targeted at middle- and high-school scenarios.
- **Deployment.** Adapted to heterogeneous camera systems in primary and secondary schools, with real-time alerts pushed to school staff.

## Demo: detection + tracking stage

`demo/track_demo.py` is a self-contained YOLO + DeepSORT pipeline — people are detected per frame and associated into stable track IDs:

```bash
pip install -r requirements.txt
python demo/track_demo.py --source path/to/video.mp4          # video file
python demo/track_demo.py --source 0                          # webcam
python demo/track_demo.py --source in.mp4 --save out.mp4 --no-show
```

The first run downloads YOLO weights (~18 MB) automatically. The behavior-recognition stage and school-specific training are not included (see provenance note).
