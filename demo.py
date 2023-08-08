import soundfile as sf
import base64
import requests
import json
import librosa
import pyaudio
import numpy as np


def convert_chunk(wav_input, sr, target_sr=22050, speaker='ahmadCorrect', api_url = "https://scevcplusplus.ngrok.app"):
    """
    Inputs:
        wav_input:
        Speaker: can be one of the following: ahmadCorrect, obama, sundar, modi, emma, priyanka, ravish, aubrey, lex, oprah, miley
    Outputs:
        response_wav: numpy array output cloned voice file
    """
    if len(wav_input.shape) > 1: # wav file not mono
        wav = wav_input[:, 0]
    else:
        wav = wav_input.copy()
    

    # Resample to 22050
    if sr != target_sr:
        wav = librosa.resample(y=wav, orig_sr=sr, target_sr=target_sr) # Can save time by avoiding this if audio is already 22050
    
    wav_base64 = base64.b64encode(wav)

    response = requests.post(f"{api_url}/convert_voice", data={"audio": wav_base64, "speaker" : speaker} )
    response_data = json.loads(response.text)
    response_wav = base64.b64decode(response_data['audio_converted'])
    response_chunk = np.frombuffer(response_wav, dtype=np.float32)
    
    return response_chunk


def select_microphone():
    p = pyaudio.PyAudio()
    info = p.get_host_api_info_by_index(0)
    num_devices = info.get('deviceCount')
    
    print("Available microphones:")
    for i in range(num_devices):
        device_info = p.get_device_info_by_host_api_device_index(0, i)
        if 'microphone' in device_info['name'].lower():
            print(f"{i}: {device_info['name']}")

    device_index = int(input("Enter the device index of the microphone you want to use: "))
    return device_index

def test_microphone(device_index):
    chunk_size = 1024
    sample_rate = 44100
    p = pyaudio.PyAudio()
    
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=sample_rate,
                    input=True,
                    output=True,
                    frames_per_buffer=chunk_size,
                    input_device_index=device_index,
                    output_device_index=device_index)
    
    print("Testing microphone...")
    
    for _ in range(int(sample_rate / chunk_size * 4)):
        data = stream.read(chunk_size)
        stream.write(data)
    
    stream.stop_stream()
    stream.close()
    p.terminate()

def record_microphone(device_index, rate, duration):
    chunk_size = int(rate*duration)
    sample_rate = rate
    p = pyaudio.PyAudio()
    
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=sample_rate,
                    input=True,
                    frames_per_buffer=chunk_size,
                    input_device_index=device_index)
    
    print("Recording started...")
    
    while True:
        data = stream.read(chunk_size)
        audio_chunk = np.frombuffer(data, dtype=np.int16)
        yield audio_chunk
    else:
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    selected_device_index = select_microphone()
    test_microphone(selected_device_index)
    
    target_rate = 44100  # Adjust this as needed
    audio_generator = record_microphone(selected_device_index, target_rate)
    
    for _ in range(int(target_rate * 4 / 44100)):  # Recording for 4 seconds
        audio_chunk = next(audio_generator)
        # Process the audio chunk as needed
