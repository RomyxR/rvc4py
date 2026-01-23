from setuptools import setup

setup(
    name="rvc4py",
    version="0.0.1",
    description="Simple RVC inference in Python",
    python_requires=">=3.10",
    license="MIT",
    install_requires=[
    'numpy<2.0',
    'soundfile==0.12.1',
    'librosa==0.10.1',
    'praat-parselmouth==0.4.3',
    'pyworld==0.3.4',
    'torchcrepe==0.0.22',
    'av>=10.0.0',
    'faiss-cpu==1.7.4',
    'python-dotenv==1.0.0',
    'pydub==0.25.1',
    'click==8.1.7',
    'tensorboardx==2.6.2.2',
    'poethepoet==0.24.4',
    'numba==0.58.1',
    'tqdm',
    'fairseq_fixed',
    'torch',
    'torchaudio',
    ],
)
