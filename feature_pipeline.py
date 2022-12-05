!add-apt-repository -y ppa:jonathonf/ffmpeg-4
!apt update
!apt install -y ffmpeg
!pip install datasets>=2.6.1
!pip install git+https://github.com/huggingface/transformers
!pip install librosa
!pip install evaluate>=0.30
!pip install jiwer
!pip install gradio
!pip install hopsworks

from huggingface_hub import login
login("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",True) 
import hopsworks
project = hopsworks.login()   # install lib and login Hopsworks and Huggingface.

from datasets import load_dataset, DatasetDict
common_voice = DatasetDict()
common_voice["train"] = load_dataset("mozilla-foundation/common_voice_11_0", "sv-SE", split="train+validation", use_auth_token=True)
common_voice["test"] = load_dataset("mozilla-foundation/common_voice_11_0", "sv-SE", split="test", use_auth_token=True)
print(common_voice)  #download Swedish language package and split them into train and test.


common_voice = common_voice.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes"])
print(common_voice)  #remove data which are not used

from transformers import WhisperFeatureExtractor
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small") #import WhisperFeatureExtractor with small data size

from transformers import WhisperTokenizer
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="Swedish", task="transcribe") import WhisperTokenizer #import WhisperTokenizer

input_str = common_voice["train"][0]["sentence"]
labels = tokenizer(input_str).input_ids
decoded_with_special = tokenizer.decode(labels, skip_special_tokens=False)
decoded_str = tokenizer.decode(labels, skip_special_tokens=True)
print(f"Input:                 {input_str}")
print(f"Decoded w/ special:    {decoded_with_special}")
print(f"Decoded w/out special: {decoded_str}")
print(f"Are equal:             {input_str == decoded_str}") #a test for WhisperTokenizer

from transformers import WhisperProcessor
processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="Swedish", task="transcribe") #combine whisperTokenizer and WhisperFeatureExtractor to a function called WhisperProcessor

print(common_voice["train"][0])#print data[0]



from datasets import Audio
common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000)) #sampling with 16kHZ

def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]
    # compute log-Mel input features from input audio array 
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    # encode target text to label ids 
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch #prepare the data with id and add labels
  
  common_voice = common_voice.map(prepare_dataset, remove_columns=common_voice.column_names["train"], num_proc=1)
  common_voice.save_to_disk("common_voice") #save data to colab
  
  cc = DatasetDict.load_from_disk("common_voice")
  
  import os
print(os.getcwd())
print(os.listdir("./common_voice/"))
print(os.listdir("./common_voice/train"))
print(os.listdir("./common_voice/test"))

def get_dir_size(path="/content/common_voice/train"):
    total = 0
    with os.scandir(path) as it:
        for entry in it:
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_dir_size(entry.path)
    return total

sz = get_dir_size(path="./common_voice/")
print(sz) #test if the file is right

common_voice.save_to_disk(F"/content/drive/My Drive/common_voice/")
print(os.listdir(F"/content/drive/My Drive/common_voice"))

cc2 = DatasetDict.load_from_disk("/content/drive/My Drive/common_voice")
cc2
cc = DatasetDict.load_from_disk("common_voice") #save the file with Google Drive


  

