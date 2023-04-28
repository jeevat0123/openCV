import cv2

# Load video capture device (default webcam)
cap = cv2.VideoCapture(0)

# Load Haar Cascade classifier for detecting faces
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize counter
counter = 0

while True:
    # Read frame from video capture device
    ret, frame = cap.read()
    
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame using Haar Cascade classifier
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        
    # Loop over the detected faces and draw bounding boxes
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 1, 2), 3)
    
    # Update counter with the number of detected faces
    counter = len(faces)
    
    # Display the resulting frame with counter
    cv2.putText(frame, "Number of people: {}".format(counter), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('frame', frame)
    
    # Exit loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture device and destroy all windows
cap.release()
cv2.destroyAllWindows()
