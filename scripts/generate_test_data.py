"""
Script to create dummy videos for testing alignment.
"""

import cv2
import numpy as np
import os

def create_moving_rect_video(filename, width=640, height=480, fps=30, duration=5, speed=5):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))

    frame_count = int(fps * duration)

    # Object properties
    rect_w, rect_h = 50, 50
    rect_y = height // 2 - rect_h // 2
    rect_x = 0

    # Texture for the rectangle (random noise) to give features
    texture = np.random.randint(0, 255, (rect_h, rect_w), dtype=np.uint8)
    texture_bgr = cv2.cvtColor(texture, cv2.COLOR_GRAY2BGR)

    for i in range(frame_count):
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Draw background features (static stars)
        for j in range(0, width, 50):
            cv2.circle(frame, (j, 100), 2, (255, 255, 255), -1)
            cv2.circle(frame, (j, 300), 2, (255, 255, 255), -1)

        # Draw moving object
        current_x = int(rect_x)
        if current_x + rect_w < width:
            # ROI
            roi = frame[rect_y:rect_y+rect_h, current_x:current_x+rect_w]
            # Overlay texture
            frame[rect_y:rect_y+rect_h, current_x:current_x+rect_w] = texture_bgr

        out.write(frame)
        rect_x += speed

    out.release()
    print(f"Created {filename}")

def main():
    dataset_dir = "./data"
    plan_dir = os.path.join(dataset_dir, "PlanTest")

    if not os.path.exists(plan_dir):
        os.makedirs(plan_dir)

    v1_path = os.path.join(plan_dir, "video1.mp4")
    v2_path = os.path.join(plan_dir, "video2.mp4")

    # Video 1: Normal speed
    create_moving_rect_video(v1_path, speed=5, duration=5)

    # Video 2: Slower speed (so it takes longer to traverse)
    # Start at same position.
    create_moving_rect_video(v2_path, speed=5, duration=8)

if __name__ == "__main__":
    main()
