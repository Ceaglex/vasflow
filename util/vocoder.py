# nohup python vocoder.py > vocoder.out 2>&1 &

from dataset.video import VideoDataset, collate_fn_for_text
from types import SimpleNamespace
from torch.utils.data import DataLoader
from util.mel_filter import extract_batch_mel
from transformers import SpeechT5HifiGan
from diffusers.models import AutoencoderKL
import torchaudio
from tqdm import tqdm

audio_process = {
   "duration":10.0,
   "sample_rate":16000,
   "channel_num":1
}
audio_process = SimpleNamespace(**audio_process)
video_process = {
   "duration":10.0,
   "resize_input_size":[224, 224],
   "target_sampling_rate":10,
   "raw_duration_min_threshold":0.2,
}
video_process = SimpleNamespace(**video_process)
dataset = VideoDataset(
   meta_dir="/home/chengxin/chengxin/Dataset_Sound/MetaData/vaflow2_meta/meta",
   dilimeter="|",
   split="test_25_ref_Chem_LRS2_GRID_LJSpeech",
   load_mode_item="video_feat_ref_text_with_waveform",
   verbose=False,
   audio_process_config=audio_process,
   video_process_config=video_process
)
dataloader = DataLoader(
    dataset,
    batch_size=16,
    num_workers=0,
    prefetch_factor=None,
    collate_fn=collate_fn_for_text,
    persistent_workers=False
)




vae = AutoencoderKL.from_pretrained(
    # From pretrained
    "/data-04/xihua/data/ckpt/audioldm2/huggingface/vae",
    local_files_only=True,
    scaling_factor=1,
    low_cpu_mem_usage=False, 
    ignore_mismatched_sizes=False,
    use_safetensors=True,
)
vocoder = SpeechT5HifiGan.from_pretrained(
    # From pretrained
    "/data-04/xihua/data/ckpt/audioldm2/huggingface/vocoder",
    local_files_only=True,
    low_cpu_mem_usage=True, 
    ignore_mismatched_sizes=False,
    use_safetensors=True,
)



for batch in tqdm(dataloader):
    audio_waveform = batch["audio_waveform"]
    audio_waveform, log_mel_spec = extract_batch_mel(audio_waveform.squeeze(1), cut_audio_duration = 10, sampling_rate = 16000, hop_length = 160, maximum_amplitude = 0.5,
                                                        filter_length = 1024, n_mel = 64, mel_fmin = 0, mel_fmax = 8000, win_length = 1024)
    log_mel_spec = log_mel_spec.unsqueeze(1)  # [bs, 1, target_mel_length (sr*duration/hop_length, 1000), n_mel (64)]
    
    audio_latent = vae.encode(log_mel_spec.to(vae.encoder.conv_in.weight.dtype)).latent_dist    #  [bs, 8, target_mel_length/4(250), n_mel/4(16)]
    audio_latent = audio_latent.sample()

    mel_spectrogram = vae.decode(audio_latent).sample                             # [bs, 1, target_mel_length(latent_length*4), 64(16*4)]
    gen_audio = vocoder(mel_spectrogram.squeeze(1))                               # [bs, duration*sr+...]

    for i in range(gen_audio.shape[0]):
        video_id = batch['video_path'][i].split("/")[-1]
        save_path = f"/home/chengxin/chengxin/vasflow/log/vae_vocoder/{video_id}"
        torchaudio.save(save_path, gen_audio[i:i+1], sample_rate=16000)
        print(video_id)
    