import os
import torch
from torch.hub import download_url_to_file
import fairseq.data.dictionary
from scipy.io import wavfile
torch.serialization.add_safe_globals([fairseq.data.dictionary.Dictionary])
from .rvc.modules.vc.modules import VC

HUBERT_URL = "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt"
RMVPE_URL = "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt"

def download_model(url, destination):
    if not os.path.exists(destination):
        print(f"Downloading {os.path.basename(destination)}...")
        try:
            download_url_to_file(url, destination, progress=True)
            print(f"Successfully downloaded to {destination}")
        except Exception as e:
            print(f"Error downloading {url}: {e}")
            raise

def rvc_voice_conversion(
        input_file: str,
        output_file: str,
        model_path: str,
        index_path: str | None = None,
        f0_method: str = "rmvpe",
        sid: int = 0,
        f0_up_key: int = 0,
        index_rate: float = 0.75,
        filter_radius: int = 3,
        resample_sr: int = 0,
        rms_mix_rate: float = 0.25,
        protect: float = 0.33,
        hubert_path: str = "./",
        rmvpe_path = "./",
        ) -> str:
    
    
    # Преобразование голоса с помощью RVC
    
    model = os.path.basename(model_path)
    weight_root = os.path.dirname(model_path)

    if index_path is None or not os.path.exists(index_path):
        index_root = ""
    else:
        index_root = os.path.dirname(index_path)

    # Загрузка hubert
    if not os.path.isfile(hubert_path):
        target_dir = os.path.dirname(hubert_path) or "."
        target_file = os.path.join(target_dir, "hubert_base.pt")
        download_model(HUBERT_URL, target_file)
        hubert_path = target_file

    # Загрузка rmvpe
    if f0_method == "rvmpe":
        if not os.path.isfile(rmvpe_path):
            target_dir = os.path.dirname(rmvpe_path) or "."
            target_file = os.path.join(target_dir, "rmvpe.pt")
            download_model(RMVPE_URL, target_file)
            rmvpe_path = target_file


    os.environ["weight_root"] = weight_root
    os.environ["index_root"] = index_root
    os.environ["hubert_path"] = hubert_path
    os.environ["rmvpe_path"] = rmvpe_path
    os.environ["rmvpe_root"] = os.path.dirname(rmvpe_path) if os.path.dirname(rmvpe_path) else "."
    
    vc = VC()
    vc.get_vc(model)
    # Вызов инференса
    tgt_sr, audio_opt, times, _ = vc.vc_inference(
        sid, # sid
        input_file, # input_path
        f0_up_key, # f0_up_key
        f0_method, # f0_method
        None, # f0_file
        index_path, # file_index
        index_rate, # index_rate
        filter_radius, # filter_radius
        resample_sr, # resample_sr
        rms_mix_rate, # rms_mix_rate
        protect, # protect
    )
    # Сохранение результата
    wavfile.write(output_file, tgt_sr, audio_opt)
    return os.path.abspath(output_file)
