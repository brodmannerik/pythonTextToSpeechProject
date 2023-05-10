# requirements
# pip install TTS

# necessary modules
from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer

path = "/home/erik/PycharmProjects/pythonTextToSpeechProject/venv/lib/python3.8/site-packages/TTS/.models.json"

model_manager = ModelManager(path)

model_path, config_path, model_item = model_manager.download_model("tts_models/en/ljspeech/tacotron2-DDC")

voc_path, voc_config_path, _ = model_manager.download_model(model_item["default_vocoder"])

syn = Synthesizer(
    tts_checkpoint=model_path,
    tts_config_path=config_path,
    vocoder_checkpoint=voc_path,
    vocoder_config=voc_config_path
)

text = "I am Johannes Kiel"

outputs = syn.tts(text)
syn.save_wav(outputs, "audio.wav")


