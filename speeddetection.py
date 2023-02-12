import cv2
import numpy as np

# Load the video file
cap = cv2.VideoCapture("speedtest1.mp4")

# Get the frames per second (fps) of the video
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create the video writer
fourcc = cv2.VideoWriter_fourcc(*"MJPG")
out = cv2.VideoWriter("output.avi", fourcc, fps, (640, 480))

# Set the first frame as the reference frame
ret, frame1 = cap.read()
frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

while True:
    # Read the current frame
    ret, frame2 = cap.read()
    if not ret:
        break

    # Convert the current frame to grayscale
    frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Calculate the absolute difference between the current frame and the reference frame
    diff = cv2.absdiff(frame1, frame2)

    # Threshold the difference to create a binary mask
    thresh = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)[1]

    # Dilate the mask to fill any holes and make the motion region easier to see
    dilated = cv2.dilate(thresh, None, iterations=3)

    # Find contours in the binary mask
    contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Loop through the contours
    for contour in contours:
        # Ignore contours that are too small
        if cv2.contourArea(contour) < 500:
            continue

        # Draw a bounding rectangle around the motion
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Calculate the speed of the vehicle in mph
        speed = fps * 70 / w

        # Put the speed text on the frame
        cv2.putText(frame2, f"Speed: {speed:.2f} mph", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Write the output frame to the video writer
    out.write(frame2)

    # Show the output frame
    cv2.imshow("Output", frame2)

    # Update the reference frame
    frame1 = frame2

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture and video writer
cap.release()
out.release()

# Close all windows
cv2.destroyAllWindows()
