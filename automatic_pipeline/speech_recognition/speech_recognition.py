from google.cloud import speech_v1p1beta1 as speech
import subprocess
import os
import shutil

video_folder = '/home/tseren/ML/iHear/Model/data/video'
mp3_folder = '/home/tseren/ML/iHear/Model/data/mp3'
wav_folder = '/home/tseren/ML/iHear/Model/data/wav'


def extract_wav():
    try:
        shutil.rmtree(mp3_folder)
    except:
        pass

    try:
        shutil.rmtree(wav_folder)
    except:
        pass

    os.makedirs(mp3_folder)
    os.makedirs(wav_folder)
    video_list = [file for file in os.listdir(video_folder) if file[-3:] == 'mp4']
    for video in video_list: ## looks like it possible to use just one command
        # ffmpeg -i test.mp4 -ab 160k -ac 2 -ar 44100 -vn audio.wav
        convert2mp3 = '''avconv -i {} {}.mp3'''.format(os.path.join(video_folder, video),
                                                       os.path.join(mp3_folder, video[:-4]))
        subprocess.call(convert2mp3, shell=True)
        conver2wav = '''ffmpeg -i {}.mp3 {}.wav'''.format(os.path.join(mp3_folder, video[:-4]),
                                                          os.path.join(wav_folder, video[:-4]))
        subprocess.call(conver2wav, shell=True)
    print("WAV files extracted!")


def main():
    extract_wav()

    client = speech.SpeechClient()

    wav_file_list = [file for file in os.listdir(wav_folder) if file[-3:] == 'wav']

    list_result = []
    for wav_file in wav_file_list:

        speech_file = os.path.join(wav_folder, wav_file)
        print('speech_file =', speech_file)

        with open(speech_file, 'rb') as audio_file:
            content = audio_file.read()

        audio = speech.types.RecognitionAudio(content=content)

        config = speech.types.RecognitionConfig(
            encoding=speech.enums.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=44100,
            language_code='ru-RU',
            audio_channel_count=2,
            enable_separate_recognition_per_channel=True)

        response = client.recognize(config, audio)

        for i, result in enumerate(response.results):
            alternative = result.alternatives[0]
            transcript = u'Transcript: {}'.format(alternative.transcript)
            list_result.append('{} - {};'.format(wav_file, transcript))

    with open('result.txt', 'w') as f:
        for item in list_result:
            f.write(item)

class SpeechRecognition():
    def __init__(self):
        pass

    def extract_wav():
        try:
            shutil.rmtree(mp3_folder)
        except:
            pass

        try:
            shutil.rmtree(wav_folder)
        except:
            pass

        os.makedirs(mp3_folder)
        os.makedirs(wav_folder)
        video_list = [file for file in os.listdir(video_folder) if file[-3:] == 'mp4']
        for video in video_list:
            convert2mp3 = '''avconv -i {} {}.mp3'''.format(os.path.join(video_folder, video),
                                                        os.path.join(mp3_folder, video[:-4]))
            subprocess.call(convert2mp3, shell=True)
            conver2wav = '''ffmpeg -i {}.mp3 {}.wav'''.format(os.path.join(mp3_folder, video[:-4]),
                                                            os.path.join(wav_folder, video[:-4]))
            subprocess.call(conver2wav, shell=True)
        print("WAV files extracted!")
    
    
    
