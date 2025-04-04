# Bach Chorale Generator

This project uses deep learning to generate four-part chorales in the style of Johann Sebastian Bach. The model learns the complex harmonic patterns from Bach's compositions and can generate new music that mimics his style.

## 🎵 Overview

J.S. Bach was a master of harmony and counterpoint. This project aims to capture some of his musical genius through a neural network that:
- Learns the patterns in Bach's four-part chorales
- Generates new chorales based on a short seed sequence
- Converts the generated MIDI-like data into playable audio

## 📊 Dataset

The project uses the JSB Chorales dataset, which contains 382 chorales harmonized by Bach, encoded as MIDI note numbers for each of the four voices (soprano, alto, tenor, bass). The dataset is split into training, validation, and test sets.

## 🧠 Model Architecture

The model combines techniques from both image and sequence processing:

```
Embedding Layer (MIDI notes → vector embeddings)
↓
Dilated Causal Conv1D (kernel_size=2, filters=32)
↓
BatchNormalization
↓
Dilated Causal Conv1D (kernel_size=2, dilation_rate=2, filters=48)
↓
BatchNormalization
↓
Dilated Causal Conv1D (kernel_size=2, dilation_rate=4, filters=64)
↓
BatchNormalization
↓
Dilated Causal Conv1D (kernel_size=2, dilation_rate=8, filters=96)
↓
BatchNormalization
↓
GRU (units=256, return_sequences=True)
↓
Dense (softmax activation, predicts next note)
```

This WaveNet-inspired architecture with dilated convolutions helps the model capture patterns at different time scales, while the GRU layer helps model long-term dependencies.

## 🎼 Music Generation

The project includes two generation methods:
1. **Deterministic generation**: Always selects the most probable next note
2. **Temperature sampling**: Controls randomness/creativity in generation
   - Low temperature (0.8): More conservative, predictable compositions
   - Medium temperature (1.0): Balanced between predictable and creative
   - High temperature (1.5): More adventurous, potentially less conventional

## 🔊 Audio Synthesis

The project includes functions to convert the generated MIDI-like data into audio using sine wave synthesis:
- Converts MIDI note numbers to frequencies using the formula: `f = 440 * 2^((n-69)/12)`
- Synthesizes each note as a sine wave
- Combines the four voices and applies appropriate envelopes

## 🚀 Getting Started

### Prerequisites
- Python 3.6+
- TensorFlow 2.x
- NumPy
- Pandas
- Matplotlib
- SciPy (for audio export)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/bach-chorale-generator.git
cd bach-chorale-generator

# Install dependencies
pip install tensorflow numpy pandas matplotlib scipy

# Download the JSB Chorales dataset
# The notebook expects a file named 'jsb_chorales.tgz'
```

### Usage

Run the Jupyter notebook:
```bash
jupyter notebook Bach_Chorales.ipynb
```

## 📈 Results

The model achieves reasonable accuracy on the test set and generates chorales that follow many of Bach's compositional rules:
- Proper voice leading
- Harmonic progression
- Four-part harmony structure

Sample outputs with different temperature settings are included in the repository as WAV files:
- `bach_cold.wav`: Conservative generation (temperature=0.8)
- `bach_medium.wav`: Balanced generation (temperature=1.0)
- `bach_hot.wav`: Adventurous generation (temperature=1.5)

## 🔍 Future Improvements

- Incorporate explicit musical rules for better chord progressions
- Experiment with transformer-based architectures
- Add control over specific musical aspects (e.g., modulations, cadences)
- Create a web interface for interactive generation

## 📚 References

- [Bach Chorales Dataset](https://github.com/czhuang/JSB-Chorales-dataset)
- [WaveNet: A Generative Model for Raw Audio](https://arxiv.org/abs/1609.03499)
- [Music Generation with Deep Learning](https://arxiv.org/abs/1612.01010)

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.