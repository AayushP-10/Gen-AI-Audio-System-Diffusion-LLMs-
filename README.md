# Gen AI Audio System (Diffusion and LLMs)

[Open in Colab](https://colab.research.google.com/drive/1gy8fFGuTjdGc_KV8CBpXpJ2GhqJWSPJ_?usp=sharing)

## Diffusion-based music generation using Mel spectrograms and Hugging Face diffusers

This project explores how diffusion models can be applied to audio generation by transforming raw waveforms into Mel spectrograms and treating them as images. Models are trained to synthesize new spectrograms, which are then converted back into audio.

---

## Example Output

Automatically generated audio loop:

https://user-images.githubusercontent.com/44233095/204103172-27f25d63-5e77-40ca-91ab-d04a45d4726f.mp4

---

## How Audio Diffusion Works

Audio is converted into Mel spectrograms, which capture frequency information over time. These spectrograms are treated as images and used to train diffusion models.

The Mel class provided in the codebase supports:
- Converting audio to spectrograms with configurable resolution
- Reconstructing audio from generated spectrograms

Higher resolutions preserve more detail but require more compute.

---

## Diffusion Models

### DDPM (Denoising Diffusion Probabilistic Models)

DDPMs are trained on Mel spectrogram datasets derived from audio files. After training, the model generates new spectrograms that resemble the training distribution.

Pretrained models are available for different datasets and configurations, including latent and implicit variants.

| Model | Dataset | Description |
|-------|---------|-------------|
| [teticio/audio-diffusion-256](https://huggingface.co/teticio/audio-diffusion-256) | [teticio/audio-diffusion-256](https://huggingface.co/datasets/teticio/audio-diffusion-256) | My "liked" Spotify playlist |
| [teticio/audio-diffusion-breaks-256](https://huggingface.co/teticio/audio-diffusion-breaks-256) | [teticio/audio-diffusion-breaks-256](https://huggingface.co/datasets/teticio/audio-diffusion-breaks-256) | Samples that have been used in music, sourced from [WhoSampled](https://whosampled.com) and [YouTube](https://youtube.com) |
| [teticio/audio-diffusion-instrumental-hiphop-256](https://huggingface.co/teticio/audio-diffusion-instrumental-hiphop-256) | [teticio/audio-diffusion-instrumental-hiphop-256](https://huggingface.co/datasets/teticio/audio-diffusion-instrumental-hiphop-256) | Instrumental Hip Hop music |
| [teticio/audio-diffusion-ddim-256](https://huggingface.co/teticio/audio-diffusion-ddim-256) | [teticio/audio-diffusion-256](https://huggingface.co/datasets/teticio/audio-diffusion-256) | De-noising Diffusion Implicit Model |
| [teticio/latent-audio-diffusion-256](https://huggingface.co/teticio/latent-audio-diffusion-256) | [teticio/audio-diffusion-256](https://huggingface.co/datasets/teticio/audio-diffusion-256) | Latent Audio Diffusion model |
| [teticio/latent-audio-diffusion-ddim-256](https://huggingface.co/teticio/latent-audio-diffusion-ddim-256) | [teticio/audio-diffusion-256](https://huggingface.co/datasets/teticio/audio-diffusion-256) | Latent Audio Diffusion Implicit Model |
| [teticio/conditional-latent-audio-diffusion-512](https://huggingface.co/teticio/latent-audio-diffusion-512) | [teticio/audio-diffusion-512](https://huggingface.co/datasets/teticio/audio-diffusion-512) | Conditional Latent Audio Diffusion Model |

---

## Dataset Preparation

### Installation

From source:

git clone https://github.com/teticio/audio-diffusion.git  
cd audio-diffusion  
pip install .

From PyPI:

pip install audiodiffusion

---

### Convert Audio to Mel Spectrograms

For small-scale training on a single GPU:

python scripts/audio_to_images.py  
--resolution 64,64  
--hop_length 1024  
--input_dir path-to-audio-files  
--output_dir path-to-output-data

To generate a higher-resolution dataset and upload it:

python scripts/audio_to_images.py  
--resolution 256  
--input_dir path-to-audio-files  
--output_dir data/audio-diffusion-256  
--push_to_hub audio-diffusion-256

The default sample rate is 22050. If changed, parameters such as n_fft may need adjustment. Training and inference configurations must remain consistent.

---

## Model Training

### Local Training

accelerate launch --config_file config/accelerate_local.yaml  
scripts/train_unet.py  
--dataset_name data/audio-diffusion-64  
--hop_length 1024  
--output_dir models/ddpm-ema-audio-64  
--train_batch_size 16  
--num_epochs 100  
--gradient_accumulation_steps 1  
--learning_rate 1e-4  
--lr_warmup_steps 500  
--mixed_precision no

---

### High-Resolution Training on Consumer GPUs

accelerate launch --config_file config/accelerate_local.yaml  
scripts/train_unet.py  
--dataset_name audio-diffusion-256  
--output_dir models/audio-diffusion-256  
--num_epochs 100  
--train_batch_size 2  
--eval_batch_size 2  
--gradient_accumulation_steps 8  
--learning_rate 1e-4  
--lr_warmup_steps 500  
--mixed_precision no  
--push_to_hub True

---

### Training on SageMaker

accelerate launch --config_file config/accelerate_sagemaker.yaml  
scripts/train_unet.py  
--dataset_name audio-diffusion-256  
--output_dir models/ddpm-ema-audio-256  
--train_batch_size 16  
--num_epochs 100  
--gradient_accumulation_steps 1  
--learning_rate 1e-4  
--lr_warmup_steps 500  
--mixed_precision no

---

## DDIM (Denoising Diffusion Implicit Models)

DDIMs can be enabled during training with:

--scheduler ddim

DDIM allows faster inference using fewer sampling steps. When eta is set to zero, the generation process becomes deterministic and can be reversed to recover latent noise representations. This enables smooth interpolation between audio samples in latent space.

---

## Latent Audio Diffusion

Latent diffusion models operate on compressed representations produced by an autoencoder. This significantly reduces computational cost and improves interpolation behavior.

The autoencoder is trained separately and then used to encode spectrograms before diffusion.

### Train Autoencoder

python scripts/train_vae.py  
--dataset_name audio-diffusion-256  
--batch_size 2  
--gradient_accumulation_steps 12

---

### Train Latent Diffusion Model

accelerate launch ...  
--vae models/autoencoder-kl

---

## Conditional Audio Generation

Audio generation can be conditioned on encoded representations such as audio embeddings.

To encode audio samples:

from audiodiffusion.audio_encoder import AudioEncoder

audio_encoder = AudioEncoder.from_pretrained("audio-encoder")  
audio_encoder.encode(["path/to/audio.mp3"])

---

Encode an entire dataset:

python scripts/encode_audio  
--dataset_name audio-diffusion-256  
--out_file data/encodings.p

---

Train a conditional model:

accelerate launch ...  
--encodings data/encodings.p

During inference, encoded tensors must be provided to guide generation. An end-to-end example is available in the conditional generation notebook.
