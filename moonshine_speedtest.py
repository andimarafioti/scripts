import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torchaudio
import time
import seaborn as sns
import moonshine

def load_whisper_mlx():
    from lightning_whisper_mlx import LightningWhisperMLX
    return LightningWhisperMLX(model="base", batch_size=1, quant=None)

def load_moonshine():
    import moonshine
    model = moonshine.load_model("moonshine/base")
    tokenizer = moonshine.load_tokenizer()
    return model, tokenizer

def load_whisper():
    from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
    model_name = "openai/whisper-base.en"
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return model, processor

# Function to load and preprocess audio
def load_audio(file_path, target_length):
    waveform, sample_rate = torchaudio.load(file_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    # Trim or pad to target length
    if waveform.shape[1] > target_length:
        waveform = waveform[:, :target_length]
    elif waveform.shape[1] < target_length:
        pad_length = target_length - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, pad_length))
    
    return waveform.squeeze().numpy()

def run_inference(model_name, model, audio, **kwargs):
    start_time = time.time()
    if model_name == "whisper_mlx":
        model.transcribe(audio)
    elif model_name == "moonshine":
        pred_ids = model[0].generate(torch.tensor(audio).unsqueeze(0))
        model[1].decode_batch(pred_ids)
    elif model_name == "whisper":
        input_features = model[1](audio, sampling_rate=16000, return_tensors="pt").input_features
        input_features = input_features.to(model[0].device).to(torch.float16)
        predicted_ids = model[0].generate(input_features)
        model[1].batch_decode(predicted_ids, skip_special_tokens=True)
    return time.time() - start_time

def run_test(models, audio, num_trials=5):
    results = {name: [] for name in models.keys()}
    for _ in range(num_trials):
        for name, model in models.items():
            inference_time = run_inference(name, model, audio)
            results[name].append(inference_time)
    return {name: np.mean(times) for name, times in results.items()}

# Load models
models = {
    "whisper_mlx": load_whisper_mlx(),
    "moonshine": load_moonshine(),
    "whisper": load_whisper()
}

# Test parameters
audio_lengths = [1, 2, 5, 10, 15, 30]  # in seconds
sample_rate = 16000
audio_file = moonshine.ASSETS_DIR / 'beckett.wav'  # Replace with a real audio file path

# Run tests
results = []
for length in tqdm(audio_lengths):
    audio = load_audio(audio_file, length * sample_rate)
    test_results = run_test(models, audio)
    results.append((length, *test_results.values()))

# Plot results
lengths, *model_times = zip(*results)
model_names = list(models.keys())

plt.figure(figsize=(12, 6))
sns.set_style("whitegrid")
sns.set_palette("deep")

for i, name in enumerate(model_names):
    sns.lineplot(x=lengths, y=model_times[i], marker='o', label=name.capitalize(), linewidth=2.5)

plt.xlabel('Audio Length (seconds)')
plt.ylabel('Inference Time (seconds)')
plt.title('STT Model Performance Comparison (base model)', fontsize=16)
plt.legend(title='Model', title_fontsize='12', fontsize='10')

# Improve x-axis
plt.xticks(lengths)
plt.xlim(min(lengths) - 0.5, max(lengths) + 0.5)

# Add minor gridlines
plt.grid(True, which='minor', linestyle=':', alpha=0.4)

# Adjust layout and save
plt.tight_layout()
plt.savefig('stt_speed_test.png', dpi=200, bbox_inches='tight')
plt.show()

# Print results
print("\nResults:")
print("Audio Length (s) | " + " | ".join(f"{name.capitalize()} (s)" for name in model_names))
print("-" * (17 + 16 * len(model_names)))
for result in results:
    print(f"{result[0]:15d} | " + " | ".join(f"{time:14.4f}" for time in result[1:]))
