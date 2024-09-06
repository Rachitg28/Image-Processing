import numpy as np
import cv2
import librosa

def check(mean_area):
    if mean_area > 50:
        return 'metal'
    else:
        return 'cardboard'

def solution(audio_path):
    # Load the audio file
    y, sr = librosa.load(audio_path)

    # Calculate the Mel spectrogram
    n_fft = 2048
    hop_length = 512
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, fmax=22000)

    # Apply contour detection to find objects in the Mel spectrogram
    contours, _ = cv2.findContours(mel_spec.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Calculate the mean area of the detected objects
    object_areas = [cv2.contourArea(contour) for contour in contours]
    mean_area = np.mean(object_areas)

    # Determine the class based on mean area
    metal_threshold = 50  # Adjust this threshold as needed
    print(mean_area)
    return check(mean_area)