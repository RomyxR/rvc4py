import os
import torch
import fairseq.data.dictionary
torch.serialization.add_safe_globals([fairseq.data.dictionary.Dictionary])
from .rvc.modules.vc.modules import VC
from scipy.io import wavfile

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
        hubert_path: str = "hubert_base.pt",
        rmvpe_path = "rmvpe.pt",
        
        ):
    # Преобразование голоса с помощью RVC
    
    model = os.path.basename(model_path)
    weight_root = os.path.dirname(model_path)


    if index_path is None or not os.path.exists(index_path):
        index_root = ""
        print("index not find")
    else:
        index_root = os.path.dirname(index_path)


    os.environ["weight_root"] = weight_root
    os.environ["index_root"] = index_root
    os.environ["hubert_path"] = hubert_path
    os.environ["rmvpe_path"] = rmvpe_path
    os.environ["rmvpe_root"] = os.path.dirname(rmvpe_path) if os.path.dirname(rmvpe_path) else "."
    
    vc = VC()
    vc.get_vc(model)
    # Вызов инференса
    tgt_sr, audio_opt, times, _ = vc.vc_inference(
        sid,              # sid
        input_file,     # input_path
        f0_up_key,      # f0_up_key
        f0_method,      # f0_method
        None,           # f0_file
        index_path,          # file_index
        index_rate,     # index_rate
        filter_radius,  # filter_radius
        resample_sr,    # resample_sr
        rms_mix_rate,   # rms_mix_rate
        protect         # protect
    )
    # Сохранение результата
    wavfile.write(output_file, tgt_sr, audio_opt)
    print(f"DONE: {os.path.abspath(output_file)}")
    return os.path.abspath(output_file)
