from transformers import pipeline
import gradio as gr
import moviepy.editor as mp
from pytube import YouTube
import math

pipe = pipeline(model="Campfireman/whisper-small-hi")  # change to "your-username/the-name-you-picked"

segment_length = 25 # 25s per segment 

def download_video(url):
    print("Downloading...")
    local_file = (
        YouTube(url)
        .streams.filter(progressive=True, file_extension="mp4")
        .first()
        .download()
    )
    print("Downloaded")
    global my_clip
    global original_wav
    my_clip = mp.VideoFileClip(local_file)
    my_clip.audio.write_audiofile("AUDIO_ORIGINAL.wav")
    original_wav = mp.AudioFileClip("AUDIO_ORIGINAL.wav")
    global audio_length
    audio_length = original_wav.duration
    print("Overall audio time elapsed: "+str(audio_length))
    return local_file

def validate_youtube(url):
    #This creates a youtube object
    try:
        yt = YouTube(url)  
    except Exception:
        print("Hi the URL seems not a valid YouTube video link")
        return True
    #This will return the length of the video in sec as an int
    video_length = yt.length
    if video_length > 600:
        print("Your video is longer than 10 minutes")
        return False
    else:
        print("Your video is less than 10 minutes")
        return True

def validate_url(url):
    import validators
    if not validators.url(url):
        return True
    else:
        return False  

def audio_clipper(index, seg_total):
    my_audio = "audio_out"+str(index)+".wav"
    audio_clipped_obj = mp.AudioFileClip.copy(original_wav)
    print("Segment "+str(index)+":")
    # Clipping
    if (index > 0):
        print("Clipped: 0 ~ " + str(segment_length * (index)) + "sec")
        audio_clipped_obj = mp.AudioFileClip.cutout(audio_clipped_obj, 0, segment_length * (index))
    if (index < seg_total - 1):
        print("Clipped: " + str(segment_length * (index + 1)) + "~ " + str(audio_length) +" sec")
        audio_clipped_obj = mp.AudioFileClip.cutout(audio_clipped_obj, segment_length * (index + 1), audio_length)
    
    # Write out the temporary segment data
    mp.AudioFileClip.write_audiofile(audio_clipped_obj, my_audio)
    #audio_clipped_obj.audio.write_audiofile(my_audio)
    
    return my_audio

def transcribe(video_url):
    text = ""
    if validate_url(video_url):
        if not validate_youtube(video_url):
            return "The URL seems not for Youtube videos or the video is too long. Check out the errors in the log. "
        else:
            download_video(video_url)
    else: 
        return "Invalid URL. Please check the format of your link. "

    segment_count = math.ceil(audio_length / segment_length) 
    print("Total segments: "+str(segment_count))
    if segment_count <= 0:
        return "Corrupted Video Data! Invalid length of "+str(segment_count * 25)+" second(s)."
    else:
        for x in range(segment_count):
            audio = audio_clipper(x, segment_count)
            seg_text = pipe(audio, batch_size=512, truncation=True)["text"]
            print("Segtext: ")
            print(seg_text)
            text = text + seg_text
            
    return text



iface = gr.Interface(
    fn=transcribe, 
    inputs=gr.Textbox(label = "Enter the URL of the Youtube video clip here (without prefixes like http://):"), 
    outputs="text",
    title="Whisper Small SE",
    description="Video Swedish Transcriptior",
)

iface.launch()
