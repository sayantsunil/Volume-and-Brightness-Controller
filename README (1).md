# Volume and Brightness Controller Using Hand Gestures
This Python project allows you to control system volume and screen brightness using hand gestures captured via a webcam. It leverages MediaPipe for real-time hand tracking, PyCaw for audio control, and Screen Brightness Control for screen adjustments. It’s a cool, touchless way to interact with your PC!

### Demo
https://github.com/yourusername/hand-gesture-controller/assets/demo.gif (Insert a GIF or video link showcasing the project in action)

 ### Features
-Control system volume using just your thumb and index finger.
-Adjust screen brightness with a full hand gesture.
-Real-time webcam hand tracking using MediaPipe.
-Dynamic visual feedback with live overlays (volume & brightness bars).
-Touch-free, gesture-based interaction for enhanced accessibility and fun!

 ## How It Works
### Volume Control:
Raise only the thumb and index finger (fingers == [1,1,0,0,0]). The distance between them maps to system volume from 0% to 100%.
### Brightness Control:
Raise all five fingers (fingers == [1,1,1,1,1]). The distance between thumb and index finger adjusts brightness.
### Gesture Recognition:
Uses MediaPipe landmarks and calculates Euclidean distance between specific finger tips to control actions.

 ## Tech Stack

##### OpenCV	   -     Computer vision and webcam processing.
##### MediaPipe  -   	Real-time hand tracking.
##### NumPy      -    	Mathematical operations.
##### PyCaw	     -     System volume control.
##### ScreenBrightnessControl -  	Screen brightness adjustment.
##### math	     -     Distance calculations.

## Prerequisites
Python 3.7+
Windows OS (for brightness control to work using wmi method)
Webcam

