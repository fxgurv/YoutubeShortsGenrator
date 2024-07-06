import re
import os
import g4f
import cv2
import json
import time
import random
import requests
import numpy as np
from uuid import uuid4
from typing import List  
from moviepy.editor import *
import google.generativeai as genai
from moviepy.video.fx.all import crop
from PIL import Image, ImageDraw, ImageFont
from moviepy.video.tools.subtitles import SubtitlesClip

os.environ['GOOGLE_API_KEY'] = 'GOOGLE_API_KEY'
os.environ['ELEVENLABS_API_KEY'] = 'ELEVENLABS_API_KEY'
os.environ['ASSEMBLYAI_API_KEY'] = 'ASSEMBLYAI_API_KEY'

# Define parameters
TYPE = "Artificial intelligence"   # @param ["Stocks", "Similar", "Artificial intelligence", "Comic", "Selected"]
NICHE = "Historical Facts"  # @param {type:"string"}
LLM = "google"  # @param ["google", "gpt35_turbo", "gpt-4"]
PROMPTER = "google"  # @param ["google", "gpt35_turbo", "gpt-4"]
IMAGE_AI = "animefy"  # @param ["simurg", "prodia", "lexica", "animefy", "raava", "shonin", "v3"]
LANGUAGE = "Hindi"  # @param ["English", "Hindi", "Urdu"]
TTS = "ElevenLabs"  # @param ["Edge TTS", "ElevenLabs", "Coqui TTS"]
VOICE = "Dave"  # @param ['Rachel', 'Drew', 'Clyde', 'Paul', 'Domi', 'Dave', 'Fin', 'Sarah', 'Antoni', 'Thomas', 'Charlie', 'George', 'Emily', 'Elli', 'Callum', 'Patrick', 'Harry', 'Liam', 'Dorothy', 'Josh', 'Arnold', 'Charlotte', 'Matilda', 'Matthew', 'James', 'Joseph', 'Jeremy', 'Michael', 'Ethan', 'Gigi', 'Freya', 'Grace', 'Daniel', 'Lily', 'Serena', 'Adam', 'Nicole', 'Bill', 'Jessie', 'Sam', 'Glinda', 'Giovanni', 'Mimi', 'Claribel Dervla', 'Daisy Studious', 'Gracie Wise', 'Tammie Ema', 'Alison Dietlinde', 'Ana Florence', 'Annmarie Nele', 'Asya Anara', 'Brenda Stern', 'Gitta Nikolina', 'Henriette Usha', 'Sofia Hellen', 'Tammy Grit', 'Tanja Adelina', 'Vjollca Johnnie', 'Andrew Chipper', 'Badr Odhiambo', 'Dionisio Schuyler', 'Royston Min', 'Viktor Eka', 'Abrahan Mack', 'Adde Michal', 'Baldur Sanjin', 'Craig Gutsy', 'Damien Black', 'Gilberto Mathias', 'Ilkin Urbano', 'Kazuhiko Atallah', 'Ludvig Milivoj', 'Suad Qasim', 'Torcull Diarmuid', 'Viktor Menelaos', 'Zacharie Aimilios', 'Nova Hogarth', 'Maja Ruoho', 'Uta Obando', 'Barbora MacLean', 'Alexandra Hisakawa', 'Alma María', 'Rosemary Okafor', 'Ige Behringer', 'Filip Traverse', 'Damjan Chapman', 'Wulf Carlevaro', 'Aaron Dreschner', 'Kumar Dahl', 'Eugenio Mataracı', 'Ferran Simen', 'Xavier Hayasaka', 'Luis Moray', 'Marcos Rudaski']
DIMENTION = "Landscape" # @param ["Landscape", "Portrait", "Squre"]
FONTS = "Ubuntu"  # @param ["Ubuntu", "Times New Roman"]
SIZE = "50"  # @param {type:"string"}
COLORS = "White"  # @param ["Black", "White"]
POSITION = "Center"  # @param ["Center", "Top", "Bottom"]
INTERVAL = "02:30"  # @param {type:"string"}

# Function to parse model name
def parse_model(model_name: str) -> any:
    if model_name == "gpt4":
        return g4f.models.gpt_4
    elif model_name == "gpt-3.5-turbo":
        return g4f.models.gpt_35_turbo
    else:
        # Default model is gpt-3.5-turbo
        return g4f.models.gpt_35_turbo

# Function to generate response using g4f or Google Gemini
def generate_response(prompt: str, model: str) -> str:
    if model == "google":
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        return response.text  # Google Gemini returns a simple string
    else:
        response = g4f.ChatCompletion.create(
            model=parse_model(model),
            messages=[{"role": "user", "content": prompt}]
        )
        return response  # g4f returns a simple string

# Function to generate topic
def generate_topic(niche: str) -> str:
    prompt = f"Please generate a specific video idea that takes about the following topic: {niche}. Make it exactly one sentence. Only return the topic, nothing else."
    return generate_response(prompt, LLM)

# Call generate_topic to assign a value to subject
subject = generate_topic(NICHE)
print(f"Generated Topic: {subject}") # Now you can print subject

def generate_script(subject: str, language: str) -> str:
    prompt = f"""
    Generate a script for a video in 4 sentences, depending on the subject of the video.
    Subject: {subject}
    Language: {language}
    """
    script = generate_response(prompt, LLM)
    return re.sub(r"\*", "", script)
script = generate_script(subject, LANGUAGE)
print(f"Generated Script: {script}")

def generate_metadata(subject: str, script: str) -> dict:
    title_prompt = f"Please generate a YouTube Video Title for the following subject, including hashtags: {subject}. Only return the title, nothing else. Limit the title under 100 characters."
    description_prompt = f"Please generate a YouTube Video Description for the following script: {script}. Only return the description, nothing else."
    title = generate_response(title_prompt, LLM)
    description = generate_response(description_prompt, LLM)
    return {"title": title, "description": description}

metadata = generate_metadata(subject, script)
print(f"Generated Metadata: {metadata}")

# Function to generate prompts
def generate_prompts(subject: str, script: str) -> List[str]:
    prompt = f"""
    Generate 4 Image Prompts for AI Image Generation, based on:
    Subject: {subject}
    Script: {script}
    Return as a JSON array of strings.
    """
    completion = generate_response(prompt, PROMPTER)

    # Clean the response to ensure it's valid JSON
    completion = completion.strip().replace("```json", "").replace("```", "")

    # Check if completion is a valid JSON string before decoding
    if completion:
        try:
            return json.loads(completion)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            print(f"Raw response: {completion}")
            return []
    else:
        print("Warning: Received empty response from the model.")
        return []
image_prompts = generate_prompts(subject, script)
print(f"Generated Image Prompts: {image_prompts}")

def generate_image(prompt: str) -> str:
    url = f"https://hercai.onrender.com/{IMAGE_AI}/text2image?prompt={prompt}"
    r = requests.get(url)

    # Check the response status code
    if r.status_code == 200:
        try:
            # Attempt to decode JSON response
            response_data = r.json()
            image_url = response_data.get("url")  # Use .get() to avoid KeyError if 'url' is missing
            if image_url:
                image_path = f"/content/{str(uuid4())}.png"
                with open(image_path, "wb") as image_file:
                    image_file.write(requests.get(image_url).content)
                return image_path
            else:
                print("Error: 'url' key not found in JSON response.")
                return None
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            print(f"Raw response: {r.text}")
            return None
    else:
        print(f"Error: Request failed with status code {r.status_code}")
        print(f"Raw response: {r.text}")
        return None

images = [generate_image(prompt) for prompt in image_prompts]
print(f"Generated Images: {images}")

def get_elevenlabs_voices():
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        raise ValueError("ELEVENLABS_API_KEY environment variable not set")

    url = "https://api.elevenlabs.io/v1/voices"
    headers = {
        "Accept": "application/json",
        "xi-api-key": api_key
    }

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        voices = response.json()["voices"]
        return {voice["name"]: voice["voice_id"] for voice in voices}
    else:
        print(f"Error: Request failed with status code {response.status_code}")
        print(f"Raw response: {response.text}")
        return None

def generate_audio(script: str, voice: str, language: str) -> str:
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        raise ValueError("ELEVENLABS_API_KEY environment variable not set")

    voices = get_elevenlabs_voices()
    if not voices:
        raise ValueError("Failed to fetch available voices")

    if voice not in voices:
        print(f"Warning: Voice '{voice}' not found. Using the first available voice.")
        voice_id = next(iter(voices.values()))
    else:
        voice_id = voices[voice]

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": api_key
    }
    data = {
        "text": script,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.5
        }
    }

    response = requests.post(url, json=data, headers=headers)

    if response.status_code == 200:
        audio_path = f"/content/{str(uuid4())}.mp3"
        with open(audio_path, "wb") as audio_file:
            audio_file.write(response.content)
        return audio_path
    else:
        print(f"Error: Request failed with status code {response.status_code}")
        print(f"Raw response: {response.text}")
        return None

# Generate audio using ElevenLabs
audio_path = generate_audio(script, VOICE, LANGUAGE)
print(f"Generated Audio: {audio_path}")

# Function to generate subtitles using AssemblyAI
def generate_subtitles(audio_path: str) -> str:
    api_key = os.getenv("ASSEMBLYAI_API_KEY")
    headers = {
        "authorization": api_key,
        "content-type": "application/json"
    }
    upload_url = "https://api.assemblyai.com/v2/upload"
    transcript_url = "https://api.assemblyai.com/v2/transcript"

    # Upload audio file
    with open(audio_path, 'rb') as f:
        response = requests.post(upload_url, headers=headers, files={'file': f})
    if response.status_code != 200:
        raise Exception(f"Failed to upload audio file: {response.text}")
    audio_url = response.json()['upload_url']

    # Request transcription
    transcript_request = {
        "audio_url": audio_url
    }
    response = requests.post(transcript_url, headers=headers, json=transcript_request)
    if response.status_code != 200:
        raise Exception(f"Failed to request transcription: {response.text}")
    transcript_id = response.json()['id']

    # Wait for transcription to complete
    while True:
        response = requests.get(f"{transcript_url}/{transcript_id}", headers=headers)
        if response.status_code != 200:
            raise Exception(f"Failed to get transcription status: {response.text}")
        status = response.json()['status']
        if status == 'completed':
            break
        elif status == 'failed':
            raise Exception("Transcription failed")
        time.sleep(5)

    # Get subtitles
    response = requests.get(f"{transcript_url}/{transcript_id}/srt", headers=headers)
    if response.status_code != 200:
        raise Exception(f"Failed to get subtitles: {response.text}")
    subtitles = response.text

    srt_path = f"/content/{str(uuid4())}.srt"
    with open(srt_path, "w") as file:
        file.write(subtitles)
    return srt_path

# Generate subtitles
subtitles_path = generate_subtitles(audio_path)
print(f"Generated Subtitles: {subtitles_path}")

# Function to choose a random song from the "Music" folder
def choose_random_song() -> str:
    music_folder = "Music"
    songs = [os.path.join(music_folder, song) for song in os.listdir(music_folder) if song.endswith('.mp3')]
    return random.choice(songs)

# Function to get the fonts directory
def get_fonts_dir() -> str:
    return "Fonts"

# Function to create a text image using Pillow
def create_text_image(text: str, font_path: str, font_size: int, image_size: tuple, text_color: str, bg_color: str) -> str:
    if not os.path.exists(font_path):
        raise OSError(f"Font file not found: {font_path}")
    img = Image.new('RGB', image_size, color=bg_color)
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype(font_path, font_size)
    text_width, text_height = d.textsize(text, font=font)
    position = ((image_size[0] - text_width) // 2, (image_size[1] - text_height) // 2)
    d.text(position, text, font=font, fill=text_color)
    image_path = f"/content/{str(uuid4())}.png"
    img.save(image_path)
    return image_path

# Function to combine everything into the final video
def combine(images: List[str], audio_path: str, subtitles_path: str, font: str) -> str:
    combined_image_path = f"/content/{str(uuid4())}.mp4"
    tts_clip = AudioFileClip(audio_path)
    max_duration = tts_clip.duration
    req_dur = max_duration / len(images)

    # Make a generator that returns an ImageClip when called with consecutive
    generator = lambda txt: ImageClip(create_text_image(
        txt,
        font_path=os.path.join(get_fonts_dir(), font + ".ttf"),
        font_size=100,
        image_size=(1080, 1920),
        text_color="#FFFF00",
        bg_color="black"
    ))

    clips = []
    tot_dur = 0
    while tot_dur < max_duration:
        for image_path in images:
            clip = ImageClip(image_path)
            clip.duration = req_dur
            clip = clip.set_fps(30)

            if round((clip.w/clip.h), 4) < 0.5625:
                clip = crop(clip, width=clip.w, height=round(clip.w/0.5625), x_center=clip.w / 2, y_center=clip.h / 2)
            else:
                clip = crop(clip, width=round(0.5625*clip.h), height=clip.h, x_center=clip.w / 2, y_center=clip.h / 2)
            clip = clip.resize((1080, 1920))

            clips.append(clip)
            tot_dur += clip.duration

    final_clip = concatenate_videoclips(clips)
    final_clip = final_clip.set_fps(30)
    random_song = choose_random_song()

    # Burn the subtitles into the video
    subtitles = SubtitlesClip(subtitles_path, generator)
    subtitles.set_pos(("center", "center"))
    random_song_clip = AudioFileClip(random_song).set_fps(44100)
    random_song_clip = random_song_clip.fx(afx.volumex, 0.1)
    comp_audio = CompositeAudioClip([tts_clip.set_fps(44100), random_song_clip])

    final_clip = final_clip.set_audio(comp_audio)
    final_clip = final_clip.set_duration(tts_clip.duration)
    final_clip = CompositeVideoClip([final_clip, subtitles])

    final_clip.write_videofile(combined_image_path, threads=4)

    return combined_image_path

# Combine everything into the final video
final_video_path = combine(images, audio_path, subtitles_path, FONTS)
print(f"Generated Final Video: {final_video_path}")
