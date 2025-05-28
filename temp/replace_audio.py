import os
import subprocess
from moviepy.editor import VideoFileClip
from pathlib import Path
from tqdm import tqdm

def replace_audio_in_video(video_path, audio_path, output_path):
    """
    批量替换MP4文件的音频
    video_folder: MP4文件所在文件夹
    audio_folder: WAV文件所在文件夹
    output_folder: 输出文件保存文件夹
    """
    try:
        cmd = [
            'ffmpeg',
            '-y',
            '-i', video_path,           # 输入视频文件
            '-i', audio_path,          # 输入音频文件
            '-c:v', 'copy',            # 直接复制视频流
            '-c:a', 'aac',             # 编码音频为AAC
            '-map', '0:v:0',           # 选择视频流的第0个视频轨道
            '-map', '1:a:0',           # 选择音频流的第0个音频轨道
            '-shortest',               # 以最短的流长度为准
            output_path
        ]
            
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"已处理: {output_path}")
            
    except subprocess.CalledProcessError as e:
        print(f"处理 {output_path} 时出错: {e}")
        

import shutil
from tqdm import tqdm
from glob import glob
import re

model = 'gt_vocoder'
# epoch = '0129'
# files = glob(f"/home/chengxin/chengxin/vasflow/log/2025_05_13-11_*-vaflow_sda_dit_noise_text_mel_10l_cc_scratch/val/video/epoch_{epoch}*/speech_*_00.wav")
files = glob("/home/chengxin/chengxin/vasflow/log/vae_vocoder/*.wav")
meta_file_dir = "/home/chengxin/chengxin/Dataset_Sound/Chem"


ljfiles = []
gridfiles = []
chemfiles = []
lrsfiles = []
for audio_file in tqdm(files):
    name = audio_file.split('/')[-1][7:-7]
    # name = audio_file.split('/')[-1]
    if name.startswith('LJ00'):
        ljfiles.append(audio_file)
    elif bool(re.match(r'^s(?:0[0-9]|[12][0-9]|3[0-5])', name)):
        gridfiles.append(audio_file)
    elif name.startswith('chem'):
        chemfiles.append(audio_file)
    else:
        lrsfiles.append(audio_file)
    # target_file = f"/home/chengxin/chengxin/Dataset_Sound/VGGSound/generated_audios/{model}/{group}/{name}.wav"
    # shutil.copy(audio_file, target_file)




gen_file_paths = chemfiles
save_dir = f'{meta_file_dir}/results/{model}/data'
os.makedirs(save_dir, exist_ok=True)
for file_path in tqdm(gen_file_paths):
    file_name = file_path.split("/")[-1]
    file_id = '_'.join(file_name.split("_")[1:-1])
    sample_id = int(file_name.split("_")[-1][:-4])
    # file_id = file_name[:-4]
    # sample_id = 0

    name = f"{file_id}_sample{sample_id}"
    shutil.copy(file_path, f"{save_dir}/{name}.wav")

    gt_video_path = f"{meta_file_dir}/sentence_video_25fps/{file_id[8:]}.mp4"  # Chem
    shutil.copy(gt_video_path, f"{save_dir}/{name}.mp4")
    replace_audio_in_video(gt_video_path, f"{save_dir}/{name}.wav", f"{save_dir}/{name}.mp4")


output_scp_path = f"{meta_file_dir}/results/{model}/wer/wav.scp"
with open(output_scp_path, "w") as scp_file:
    for file_path in gen_file_paths:
        file_name = file_path.split("/")[-1]
        file_id = '_'.join(file_name.split("_")[1:-1])
        sample_id = int(file_name.split("_")[-1][:-4])

        name = f"visual_tts_{file_id}_sample{sample_id}"
        gen_path = file_path
        scp_file.write(f"{name} {gen_path}\n")

print(f"Successfully written {len(gen_file_paths)} entries to {output_scp_path}")