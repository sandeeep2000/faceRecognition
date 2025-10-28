import cv2

print("Attempting to access camera...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Failed to open camera.")
else:
    print("✅ Camera access granted.")
    cap.release()
