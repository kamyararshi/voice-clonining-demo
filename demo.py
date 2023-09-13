import soundfile as sf
import base64
import requests
import json
import librosa
import pyaudio
import numpy as np
import os

from argparse import ArgumentParser

def convert_chunk(wav_inputarray, sr, target_sr=22050, speaker='ahmadCorrect', api_url = "https://scevcplusplus.ngrok.app"):
    """
    Inputs:
        wav_input:
        Speaker: can be one of the following: ahmadCorrect, obama, sundar, modi, emma, priyanka, ravish, aubrey, lex, oprah, miley
    Outputs:
        response_wav: numpy array output cloned voice file
    """
    if len(wav_inputarray.shape) > 1: # wav file not mono
        wav = wav_inputarray[:, 0]
    else:
        wav = wav_inputarray.copy()
    

    # Resample to 22050
    if sr != target_sr:
        wav = librosa.resample(y=wav, orig_sr=sr, target_sr=target_sr) # Can save time by avoiding this if audio is already 22050
    
    wav_base64 = base64.b64encode(wav)

    response = requests.post(f"{api_url}/convert_voice", data={"audio": wav_base64, "speaker" : speaker} )
    print('#######################')
    print(response)
    print('#######################')
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
        #if 'microphone' in device_info['name'].lower():
        print(f"{i}: {device_info['name']}")
    
    device_index = int(input("Enter the device index of the microphone you want to use: "))

    print(p.get_device_info_by_index(device_index))
    return device_index

def test_microphone(device_index):
    chunk_size = 1024
    sample_rate = 44100
    p = pyaudio.PyAudio()
    
    stream = p.open(format=pyaudio.paInt16,
                    channels=1, # Set to 1 for mono microphone
                    rate=sample_rate,
                    input=True,
                    output=True,
                    frames_per_buffer=chunk_size,
                    input_device_index=device_index,
                    output_device_index=3)
    
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
        data = stream.read(chunk_size, exception_on_overflow = False)
        audio_chunk = np.frombuffer(data, dtype=np.int16)
        yield audio_chunk
    
    stream.stop_stream()
    stream.close()
    p.terminate()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--speaker', default='ahmadCorrect', choices=["ahmadCorrect", "obama", "sundar", "modi", "emma", "priyanka", "ravish", "aubrey", "lex", "oprah", "miley"])
    parser.add_argument('--target_sr', default="22050")
    parser.add_argument('--length', default="4")
    args = parser.parse_args()

    selected_device_index = select_microphone()
    test_microphone(selected_device_index)
    
    target_rate = int(args.target_sr)  # Adjust this as needed
    duration = int(args.length)
    audio_generator = record_microphone(selected_device_index, target_rate, duration=duration)
    
    #for _ in range(int(target_rate * duration / 44100)):  # Recording for 4 seconds
    
    filelist = [ f for f in os.listdir('tmp/') if f.endswith(".wav") ]
    for f in filelist:
        os.remove(os.path.join('tmp/', f))
    i=0
    while True:
        audio_chunk = next(audio_generator)
        sf.write(f'tmp/chunk{i}.wav', audio_chunk, 22050)
        wav, sr = sf.read(f'tmp/chunk{i}.wav', dtype='float32')
        # Process the audio chunk as needed
        #output_chunk = convert_chunk(audio_chunk, sr=target_rate, speaker=args.speaker)
        output_chunk = convert_chunk(wav, sr=sr, speaker=args.speaker)
        sf.write(f'tmp/output_chunk{i}.wav', output_chunk, 22050)
        i += 1