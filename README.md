# rvc4py
Simple RVC inference in Python

## 📥Installation
```bash
# Basic installation
pip install git+https://github.com/RomyxR/rvc4py.git
# Installation with GPU support
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu130
pip install git+https://github.com/RomyxR/rvc4py.git
```
## 🧩Signature
```python
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
```

## 📄Example
```python
import rvc4py

rvc4py.rvc_voice_conversion(
    "input.wav",
    "output.wav",
    r"path/to/model.pth",
    r"path/to/model.index",
    hubert_path="./",
    rmvpe_path="./",
    )
```
## ⭐Acknowledgments
**Original RVC** - https://github.com/RVC-Project/Retrieval-based-Voice-Conversion
