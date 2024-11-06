import gradio as gr
from transformers import pipeline
import whisper
import cv2
import numpy as np
import mediapipe as mp
import torch
import threading
import time

# Load Whisper model for voice transcription
device = "cuda" if torch.cuda.is_available() else "cpu"
transcribe_model = whisper.load_model("base", device=device)

# Load a pre-trained language model for skincare advice
skincare_model = pipeline("text-generation", model="distilgpt2", device=0 if device == "cuda" else -1)

# Initialize MediaPipe for facial landmark detection
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False)

# Global variable for latest frame
latest_frame = None
capture = True

# Start a new thread for capturing video
def start_video_capture():
    global latest_frame, capture
    cap = cv2.VideoCapture(0)
    while capture:
        ret, img = cap.read()
        if ret:
            latest_frame = img.copy()  # Save the raw frame
        time.sleep(0.1)  # Capture a frame every 100ms
    cap.release()

# Apply virtual makeup to a frame
def apply_virtual_makeup(image):
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for landmark in face_landmarks.landmark:
                x = int(landmark.x * image.shape[1])
                y = int(landmark.y * image.shape[0])
                cv2.circle(image, (x, y), 1, (255, 0, 0), -1)  # Simple "blush" effect
    return image

# Capture a frame with virtual makeup
def capture_and_apply_makeup():
    global latest_frame
    if latest_frame is not None:
        return apply_virtual_makeup(latest_frame)
    return None

# Transcribe voice to text using Whisper
def transcribe_voice(audio):
    transcription = transcribe_model.transcribe(audio)["text"]
    return transcription

# Generate skincare recommendation
def generate_skincare_recommendation(text_input):
    prompt = f"Generate skincare advice for the following concerns: {text_input}"
    response = skincare_model(prompt, max_length=100, num_return_sequences=1)
    return response[0]['generated_text']

# Get skincare advice based on text or voice input
def skincare_consultant(text_input=None, audio_input=None, apply_ai_makeup=False):
    if apply_ai_makeup:
        question_text = "Suggest a makeup look for a glowing, natural appearance."
    else:
        question_text = text_input or "Provide general skincare advice."
    if audio_input:
        question_text = transcribe_voice(audio_input)
    skincare_advice = generate_skincare_recommendation(question_text)
    return skincare_advice

# Start video capture in a separate thread
video_thread = threading.Thread(target=start_video_capture)
video_thread.start()

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("# Real-Time AI Makeup and Skincare Consultant")
    
    with gr.Row():
        video_feed = gr.Video( streaming=True, label="Live Video Feed")  # Use 'camera' for live feed
        capture_image = gr.Image(label="Captured Image with Makeup")

    with gr.Row():
        capture_button = gr.Button("Capture and Apply Makeup")
        capture_button.click(fn=capture_and_apply_makeup, inputs=[], outputs=capture_image)
        
    with gr.Row():
        text_input = gr.Textbox(label="Type Skincare Question (Optional)")
        audio_input = gr.Audio(type="filepath", label="Ask a Skincare Question (Optional)")
        
    skincare_advice_output = gr.Textbox(label="Skincare Advice")

    submit_button = gr.Button("Get Advice")
    submit_button.click(
        skincare_consultant,
        inputs=[text_input, audio_input, gr.Checkbox(value=False, visible=False)],  # Hidden checkbox to match inputs
        outputs=[skincare_advice_output]
    )

    # Add button for AI-recommended makeup
    ai_makeup_button = gr.Button("Apply AI Recommended Makeup")
    ai_makeup_button.click(
        skincare_consultant,
        inputs=[gr.Textbox(value="", visible=False), gr.Audio(type="filepath", visible=False), gr.Checkbox(value=True)],  # Set apply_ai_makeup to True
        outputs=[skincare_advice_output]
    )

demo.launch()

# Stop video capture when done
capture = False
video_thread.join()
