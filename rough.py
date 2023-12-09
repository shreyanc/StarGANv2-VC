
import subprocess

input_file = "input_audio.mp3"
output_file = "output_audio.mp3"
duration = 10

ffmpeg_command = f"ffmpeg -i {input_file} -ss 0 -t {duration} {output_file}"
subprocess.call(ffmpeg_command, shell=True)
