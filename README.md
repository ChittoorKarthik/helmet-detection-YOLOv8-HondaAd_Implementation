# Helmet Detection System

This project detects helmet usage in real-time video feeds. Inspired by [Honda’s “No Green Light Without Helmet” campaign](https://www.facebook.com/watch/?v=1517745148995687), it aims to promote safety awareness by highlighting individuals who are not wearing helmets.

## Features

- **Real-Time Helmet Detection**: Identifies whether individuals in a video feed are wearing helmets.
- **Separate Display for No-Helmet Instances**: Frames showing individuals without helmets are displayed on a second screen for better visibility.

## How It Works

1. **Video Processing**: The system processes two video streams simultaneously: one for helmet detection and one as a clean feed.
2. **Helmet Detection**: Using a YOLO model, the system identifies helmet and no-helmet instances in each frame.
3. **Display**: Frames with no-helmet detections are cropped and displayed on a secondary screen.
4. Also refer the PPT
