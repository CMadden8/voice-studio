# Audio engines and credits

- [Praat](https://www.fon.hum.uva.nl/praat/) and [Parselmouth](https://github.com/YannickJadoul/Parselmouth) supply conventional pitch and formant processing. The Python wheel includes its own notices. A [copy of the GPL](../licenses/Parselmouth-GPL-3.txt) is included here.
- [Rubber Band 4.0.0](https://github.com/breakfastquay/rubberband/tree/v4.0.0) supplies pitch and time stretching for unvoiced and rough input. Setup fetches the [publisher’s GPL Windows distribution](https://breakfastquay.com/rubberband/) and preserves its README and COPYING under `Runtime/rubberband/`.
- NumPy and SciPy supply numerical processing and filters. SoundFile/libsndfile handle audio files, sounddevice/PortAudio capture and play audio, and Qt for Python supplies the desktop UI. Python package versions are listed in [the lockfile](../requirements.lock.txt). Their notices are installed with the packages.
- The separately installed DaVinci Resolve Studio application and Voice Isolation effect are proprietary. The optional bridge discovers the locally installed scripting SDK. No Blackmagic binaries, models or SDK files are distributed here.

Standard cleanup estimates stationary noise from quiet windows and applies bounded spectral attenuation. Strong pitch shifts can sound artificial. Cleanup cannot recover a clipped input. Session calibration measures the reference phrase and applies corrections up to two semitones and six decibels. It lasts until restart or a microphone change.
