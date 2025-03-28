import gradio as gr
import subprocess
import time
import os
import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--listen", default="127.0.0.1", help="Listen address")
parser.add_argument("--port", type=int, default=7860, help="Port number")
parser.add_argument("--output_folder", default="./audiobook_output", help="Output folder path")
parser.add_argument("--share", action="store_true", help="Create a public link")
cmd_args = parser.parse_args()

log_path = os.path.join(cmd_args.output_folder, "e2a.log")


class Conversion:
    def __init__(self):
        self.output_folder = cmd_args.output_folder
        self.current_audiobook_path = None
        self.current_subprocess = None

    def audiobook_path(self, input_file):
        return os.path.join(self.output_folder, os.path.splitext(os.path.basename(input_file))[0])

    def start_subprocess(self, args, env):
        with open(log_path, "w") as log_file:
            self.current_subprocess = subprocess.Popen(args=args, env=env,
                                                       stdout=log_file, stderr=log_file,
                                                       bufsize=1, text=True)
        while True and self.current_subprocess is not None:
            exit_code = self.current_subprocess.poll()
            if exit_code is not None:
                print(f"Process exited with code {exit_code}")
                break
            else:
                print("Process is still running...")
                time.sleep(1)

    def stop_subprocess(self):
        if self.current_subprocess:
            self.current_subprocess.terminate()
            self.current_subprocess = None
            print("Subprocess stopped")

    def convert_epub_to_audiobook(
            self, input_file, tts, log_level, language, newline_mode, chapter_start, chapter_end,
            output_text, remove_endnotes, remove_reference_numbers,
            search_and_replace_file, worker_count, no_prompt, preview,
            azure_tts_key, azure_tts_region, openai_api_key, openai_base_url,
            voice_name, model_name, output_format, break_duration,
            voice_rate, voice_volume, voice_pitch, proxy,
            piper_path, piper_speaker, piper_sentence_silence, piper_length_scale):

        args = [
            "python", "main.py",
            str(input_file),
            self.audiobook_path(input_file),
            "--tts", tts,
            "--log", log_level,
            "--language", language,
            "--newline_mode", newline_mode,
            "--chapter_start", str(chapter_start),
            "--chapter_end", str(chapter_end),
            "--worker_count", str(worker_count),
            "--break_duration", str(break_duration),
            "--voice_name", voice_name,
            "--output_format", output_format,
            "--model_name", model_name,
            "--search_and_replace_file", search_and_replace_file,
            "--voice_rate", voice_rate,
            "--voice_volume", voice_volume,
            "--voice_pitch", voice_pitch,
            "--proxy", proxy,
            "--no_prompt" if no_prompt else None,
            "--preview" if preview else None,
            "--output_text" if output_text else None,
            "--remove_endnotes" if remove_endnotes else None,
            "--remove_reference_numbers" if remove_reference_numbers else None,
        ]

        if tts == "piper":
            args += [
                "--piper_path", piper_path,
                "--piper_speaker", str(piper_speaker),
                "--piper_sentence_silence", str(piper_sentence_silence),
                "--piper_length_scale", str(piper_length_scale),
            ]

        args = [arg for arg in args if arg is not None]

        env = os.environ.copy()
        if tts == "azure":
            env["MS_TTS_KEY"] = azure_tts_key
            env["MS_TTS_REGION"] = azure_tts_region
        elif tts == "openai":
            env["OPENAI_API_KEY"] = openai_api_key
            env["OPENAI_BASE_URL"] = openai_base_url

        self.start_subprocess(args, env)

    def preview_book(self, input_file):
        args = [
            "python", "main.py",
            "--preview",
            str(input_file),
            self.audiobook_path(input_file)
        ]
        env = os.environ.copy()
        env["MS_TTS_KEY"] = "x"
        env["MS_TTS_REGION"] = "x"
        self.start_subprocess(args, env)
        _, total_chapters = Utils().get_progress()
        self.current_audiobook_path = self.audiobook_path(input_file)
        return total_chapters

    def list_files(self):
        if self.current_audiobook_path is None or not os.path.isdir(self.current_audiobook_path):
            return []
        return [os.path.join(self.current_audiobook_path, file) for file in os.listdir(self.current_audiobook_path)]


class Utils:
    @staticmethod
    def read_log():
        try:
            with open(log_path, "r") as log_file:
                return log_file.read()
        except FileNotFoundError:
            return "Log file not found."

    @staticmethod
    def get_progress():
        result = Utils.read_log()
        current_chapters = 0
        total_chapters = 0
        for line in result.splitlines()[::-1]:
            m = re.search(r"chapter (\d+)/(\d+)", line)
            if m:
                current_chapters, total_chapters = int(m.group(1)), int(m.group(2))
                break
        return current_chapters, total_chapters


conversion = Conversion()
utils = Utils()

with gr.Blocks() as ui:
    with gr.Row():
        input_file = gr.File(label="Input EPUB File", file_types=[".epub"])
        tts = gr.Dropdown(choices=["azure", "openai", "piper"], label="TTS Provider", value="azure")
        language = gr.Textbox(label="Language", value="en-US")
        log = gr.Dropdown(choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], label="Log Level", value="INFO")

    with gr.Row():
        newline_mode = gr.Radio(choices=["single", "double", "none"], label="Newline Mode", value="double")
        chapter_start = gr.Number(label="Chapter Start Index", value=1, precision=1)
        chapter_end = gr.Number(label="Chapter End Index", value=-1, precision=1)
        worker_count = gr.Number(label="Worker Count", value=1, precision=0)

    with gr.Row():
        output_text = gr.Checkbox(label="Output Text", value=False)
        remove_endnotes = gr.Checkbox(label="Remove Endnotes", value=False)
        remove_reference_numbers = gr.Checkbox(label="Remove Reference Numbers", value=False)
        no_prompt = gr.Checkbox(label="Skip Cost Prompt", value=False)
        preview = gr.Checkbox(label="Preview Mode", value=False)

    search_and_replace_file = gr.Textbox(label="Search & Replace File Path", value="")
    output_folder = gr.Textbox(label="Output Folder", value=cmd_args.output_folder, interactive=False)

    with gr.Tab("TTS Options"):
        voice_name = gr.Textbox(label="Voice Name", value="")
        model_name = gr.Textbox(label="Model Name", value="")
        output_format = gr.Textbox(label="Output Format", value="")

    with gr.Tab("Azure/Edge Specific"):
        azure_tts_key = gr.Textbox(label="Azure TTS Key", value="")
        azure_tts_region = gr.Textbox(label="Azure TTS Region", value="")
        break_duration = gr.Textbox(label="Break Duration (ms)", value="1250")
        voice_rate = gr.Textbox(label="Voice Rate", value="")
        voice_volume = gr.Textbox(label="Voice Volume", value="")
        voice_pitch = gr.Textbox(label="Voice Pitch", value="")
        proxy = gr.Textbox(label="Proxy", value="")

    with gr.Tab("OpenAI Specific"):
        openai_api_key = gr.Textbox(label="OpenAI API Key", value="")
        openai_base_url = gr.Textbox(label="OpenAI Base URL", value="")

    with gr.Tab("Piper TTS"):
        piper_path = gr.Textbox(label="Piper Executable Path", value="piper")
        piper_speaker = gr.Number(label="Speaker ID", value=0, precision=0)
        piper_sentence_silence = gr.Number(label="Silence Between Sentences (sec)", value=0.2)
        piper_length_scale = gr.Number(label="Speaking Rate (Length Scale)", value=1.0)

    with gr.Row():
        submit_button = gr.Button("Convert to Audiobook", variant="primary")
        stop_button = gr.Button("Stop", variant="stop")

    log_textarea = gr.TextArea(label="Log", interactive=False, lines=10)
    file_list = gr.File(label="Download Audiobook", file_count="multiple", interactive=False)

    input_file.upload(conversion.preview_book, inputs=[input_file], outputs=[chapter_end])

    submit_button.click(
        conversion.convert_epub_to_audiobook,
        inputs=[
            input_file, tts, log, language, newline_mode, chapter_start, chapter_end,
            output_text, remove_endnotes, remove_reference_numbers,
            search_and_replace_file, worker_count, no_prompt, preview,
            azure_tts_key, azure_tts_region, openai_api_key, openai_base_url,
            voice_name, model_name, output_format, break_duration,
            voice_rate, voice_volume, voice_pitch, proxy,
            piper_path, piper_speaker, piper_sentence_silence, piper_length_scale
        ],
        outputs=[],
    )

    stop_button.click(conversion.stop_subprocess)
    ui.load(utils.read_log, outputs=log_textarea, every=1)
    ui.load(conversion.list_files, outputs=[file_list], every=1)

ui.queue().launch(server_name=cmd_args.listen, server_port=cmd_args.port, share=cmd_args.share)
