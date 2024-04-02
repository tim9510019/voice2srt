import torch
import os
import gradio as gr
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    pipeline,
)
import time

model_name_or_path = "openai/whisper-large-v3"
task = "automatic-speech-recognition"

chunkSec = 30
srtFolder = "srtResult"

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_name_or_path,
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=True,
    use_safetensors=True,
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_name_or_path)

pipe = pipeline(
    task,
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=20,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
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
        )["chunks"]

        convert2srt(scriptList)

    return scriptList


iface = gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(sources=["microphone", "upload"], type="filepath"),
    outputs="text",
    title="INT16 Whisper Large V3",
    description="Realtime demo for multiple language speech recognition using INT16 Whisper Large V3",
)

iface.launch(share=True)
