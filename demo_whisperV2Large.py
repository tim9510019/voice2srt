import torch
import os
import gradio as gr
from transformers import (
    AutomaticSpeechRecognitionPipeline,
    WhisperForConditionalGeneration,
    WhisperTokenizer,
    WhisperProcessor,
)
from peft import PeftModel, PeftConfig
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
model_name_or_path = "openai/whisper-large-v2"
language = "chinese"
language_abbr = "zh-TW"
task = "transcribe"
# dataset_name = "mozilla-foundation/common_voice_11_0"
# prepareCPUCore = 8

peft_model_id = "tim9510019/openai-whisper-large-v2-LORA"

chunkSec = 30
srtFolder = "srtResult"


peft_config = PeftConfig.from_pretrained(peft_model_id)
model = WhisperForConditionalGeneration.from_pretrained(
    peft_config.base_model_name_or_path, load_in_8bit=True, device_map="auto"
)
model = PeftModel.from_pretrained(model, peft_model_id)

tokenizer = WhisperTokenizer.from_pretrained(
    peft_config.base_model_name_or_path, language=language, task=task
)
processor = WhisperProcessor.from_pretrained(
    peft_config.base_model_name_or_path, language=language, task=task
)
feature_extractor = processor.feature_extractor
forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task=task)

pipe = AutomaticSpeechRecognitionPipeline(
    model=model, tokenizer=tokenizer, feature_extractor=feature_extractor
)


def convert2srt(scriptList):
    srtList = []
    for sIdx, script in enumerate(scriptList):
        secRawFrom = script["timestamp"][0]
        secRawTo = script["timestamp"][1]

        if secRawTo == None:
            secRawTo = secRawFrom + chunkSec

        content = script["text"].encode("utf-8").decode("utf-8")

        milliSecFrom = "%03d" % (int(secRawFrom * 1000) % 1000)
        milliSecTo = "%03d" % (int(secRawTo * 1000) % 1000)

        secFrom = "%02d" % (int(secRawFrom) % 60)
        secTo = "%02d" % (int(secRawTo) % 60)

        minuteFrom = "%02d" % (int(secRawFrom) // 60) % 60
        minuteTo = "%02d" % (int(secRawTo) // 60) % 60

        hourFrom = "%02d" % (int(secRawFrom) // 3600)
        hourTo = "%02d" % (int(secRawFrom) // 3600)

        srtList.append(
            f"{sIdx}\n{hourFrom}:{minuteFrom}:{secFrom},{milliSecFrom} --> {hourTo}:{minuteTo}:{secTo},{milliSecTo}\n{content}\n\n"
        )

    if not os.path.exists(srtFolder):
        os.makedirs(srtFolder)

    timeNow = time.strftime("%Y_%m_%d_%H_%M_%S", time.gmtime(time.time()))
    srtFileName = f"V2T_{timeNow}.srt"
    srtFilePath = os.path.join(srtFolder, srtFileName)

    with open(srtFilePath, "w+") as whd:
        for srt in srtList:
            print(srt, file=whd)


def transcribe(audio):
    with torch.cuda.amp.autocast():
        scriptList = pipe(
            audio,
            generate_kwargs={"forced_decoder_ids": forced_decoder_ids},
            chunk_length_s=chunkSec,
            return_timestamps=True,
        )["chunks"]

        convert2srt(scriptList)

    return scriptList


iface = gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(sources=["microphone", "upload"], type="filepath"),
    outputs="text",
    title="PEFT LoRA + INT8 Whisper Large V2",
    description="Realtime demo for Chinese speech recognition using `PEFT-LoRA+INT8` fine-tuned Whisper Large V2 model.",
)

iface.launch(share=True)
