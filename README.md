# Optical Flow Feature Tracking

This project was created as part of a **Machine Vision laboratory exercise**.  
Its goal is to detect characteristic points in a video, track them across consecutive frames using the **Lucas–Kanade optical flow** method, and visualize their motion trajectories.

## Project objective

The application:

- loads a video file provided from the command line,
- automatically detects good feature points to track,
- tracks these points over time,
- draws their current positions and motion trajectories,
- displays the processed video in real time.

## Methods used

The implementation is based on classical OpenCV techniques:

- **Shi-Tomasi corner detection** (`cv2.goodFeaturesToTrack`) for selecting feature points,
- **Pyramidal Lucas–Kanade optical flow** (`cv2.calcOpticalFlowPyrLK`) for tracking point movement between frames,
- visualization of:
  - tracked feature points,
  - accumulated motion trajectories.

## Requirements

- Python 3
- OpenCV
- NumPy

Install dependencies with:

```bash
pip install opencv-python numpy
```

## How to run

```bash
python LAB4_WMA_PL_szablon.py --video film.mp4
```

Replace `film.mp4` with the path to your own video file.

## Controls

- `q` — quit the program
- `ESC` — quit the program

## Output

The program opens a window in which:

- the video is displayed frame by frame,
- tracked points are marked,
- trajectories of their motion are drawn,
- the current number of tracked points is shown on the screen.

## Notes

This implementation focuses on **sparse optical flow**, which means that motion is tracked only for selected characteristic points instead of the whole image. This makes the solution efficient, readable, and suitable for laboratory demonstration purposes.

## Academic context

This project was prepared as part of a **Machine Vision lab assignment** focused on feature detection, point tracking, and motion visualization in video sequences.
