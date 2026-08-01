# Intelligent Monitoring System for School-Bullying Detection

Team project at Northeastern University (01/2022–09/2023), advised by Prof. Dongyue Chen. Team leader: [Wei Fu](https://barryway7.github.io)

## Overview

A real-time computer-vision system that detects potentially harmful physical interactions among students in crowded school environments and alerts staff so they can intervene promptly.

## System design

- **Detection & tracking.** YOLOv5 detects individuals; DeepSORT associates detections across frames into a real-time multi-target tracking pipeline that remains stable in crowded scenes.
- **Behavior recognition.** The tracker output feeds an interaction classifier trained to flag violent or harmful behaviors.
- **Training data.** A public violence dataset augmented with data collected and curated by our team, targeted at middle- and high-school scenarios.
- **Deployment.** Adapted to heterogeneous camera systems in primary and secondary schools, with real-time alerts pushed to school staff.

## Status

Code cleanup for public release is in progress. Contact me if you would like more details in the meantime.
