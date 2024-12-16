import cv2
import streamlit as st
from gtts import gTTS
from tempfile import NamedTemporaryFile

# Page configuration
st.set_page_config(page_title="Midori Eco", layout="wide", page_icon="♻️")

# Recycling bin classification based on detected objects
BIN_GUIDE = {
    "book": "Blue recycling bin",
    "bottle": "Red recycling bin",
    "chair": "Black recycling bin",
    "pottedplant": "Pink recycling bin",
    "tvmonitor": "Purple recycle bin"
}

# Load MobileNet SSD model
@st.cache_resource
def load_model():
    prototxt_path = "deploy.prototxt"
    model_path = "mobilenet_iter_73000.caffemodel"
    return cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

model = load_model()

# Class labels for the MobileNet SSD model

CLASS_LABELS = {
    5: "bottle",
    9: "chair",
    16: "pottedplant",
    20: "tvmonitor",
}

def play_audio(text):
    """Generate and play audio for a given text."""
    tts = gTTS(text=text, lang="en", slow=False)
    temp_audio = NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_audio.name)
    st.audio(temp_audio.name, format="audio/mp3")

def detect_objects(frame, model):
    """Detect objects using the MobileNet SSD model."""
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
    model.setInput(blob)
    detections = model.forward()
    return detections

def main():
    st.title("♻️ Midori Eco: Your Recycling Assistant")
    st.write(
        "Hello, I am **Midori Eco**, your smart assistant for recycling. "
        "Hold an item in front of the webcam and press **'Detect Object'** to find out which bin it goes in."
    )
    st.sidebar.header("User Guide")
    st.sidebar.write(
        "- **Start the webcam** to enable live detection.\n"
        "- Click **'Detect Object'** to check an item's category.\n"
        "- Press **'Stop Webcam'** when you're done."
    )

    # Start webcam feed
    video = cv2.VideoCapture(0)
    video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not video.isOpened():
        st.error("Error: Could not access the webcam.")
        return

    frame_placeholder = st.empty()  # Placeholder for displaying video frames
    detected_objects_placeholder = st.empty()
    # Placeholder for displaying detected objects
    start = st.button("Start Webcam")
    stop = st.button("Stop Webcam")  # Stop button
    detect = st.button("Detect Object")  # Button to detect the next object

    detected_labels = set()  # Keep track of already detected objects to avoid repeated audio

    while not stop:
        ret, frame = video.read()

        if not ret:
            st.error("Error: Failed to capture frame.")
            break

        # Detect objects and draw green boxes
        detections = detect_objects(frame, model)
        detected_messages = []
        h, w = frame.shape[:2]

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:  # Threshold for detection
                class_id = int(detections[0, 0, i, 1])
                label = CLASS_LABELS.get(class_id, "Unknown")
                box = detections[0, 0, i, 3:7] * [w, h, w, h]
                (startX, startY, endX, endY) = box.astype("int")

                # Draw green box for each detected object
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                text = f"{label}: {confidence:.2f}"
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                if label in BIN_GUIDE and label not in detected_labels:
                    bin_instruction = BIN_GUIDE[label]
                    message = f"I detected {label}. Please place it in the {bin_instruction}."
                    detected_messages.append(message)
                    play_audio(message)  # Play audio for the detected object
                    detected_labels.add(label)  # Add the label to the set to avoid repeats

                elif label not in detected_labels:
                    message = f"I detected {label}. I'm not sure which bin it goes in. Please check the recycling guide."
                    detected_messages.append(message)
                    play_audio(message)  # Play audio for unknown objects
                    detected_labels.add(label)  # Add the label to the set to avoid repeats

        # Display video frame and detection messages
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB")
        detected_objects_placeholder.write("\n".join(detected_messages) if detected_messages else "No objects detected.")

    # Release the video feed and play goodbye message
    video.release()
    st.subheader("Goodbye Message")
    play_audio("Thank you for using Midori Eco. Goodbye!")

if __name__ == "__main__":
    main()
