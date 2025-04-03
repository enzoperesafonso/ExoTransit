# Exoplanet Transit Animator

Generates dark-mode animations of exoplanet transits with optional embedded sonification (audio representation of flux), configured via a YAML file.

(https://github.com/enzoperesafonso/exotransit_sonification/blob/main/transit_animation_none_dark_sound.gif)

## Features

*   Simulates transit scenarios: No Limb Darkening, Quadratic Limb Darkening, LD + Spots.
*   Generates MP4 video output suitable for presentations (dark mode visuals).
*   Optional audio track embedded in the video, mapping flux to pitch (lower flux = lower pitch).
*   Most parameters configurable via `config.yaml`.
*   Command-line interface for selecting scenarios and overriding settings.
*   Phase continuity in audio generation for smoother sound.

## Requirements

*   **Python 3.7+**
*   **FFmpeg:** Essential for combining audio and video. Must be installed and accessible in your system's PATH.
    *   Download from [ffmpeg.org](https://ffmpeg.org/download.html).
    *   Ensure the `ffmpeg` command works in your terminal.
*   **Python Libraries:** See `requirements.txt`. Install using pip.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository_url> # Replace <repository_url> with your repo URL
    cd transit_animator
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # Activate it:
    # Windows: .\venv\Scripts\activate
    # macOS/Linux: source venv/bin/activate
    ```

3.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Verify FFmpeg:**
    Open your terminal/command prompt and run:
    ```bash
    ffmpeg -version
    ```
    If this command is not found, you need to install FFmpeg and add it to your system's PATH.

## Configuration (`config.yaml`)

Edit the `config.yaml` file to change simulation parameters:

*   **`system_params`:** Planet and orbital properties (passed to `batman`).
*   **`star_params`:** Star radius and rotation period.
*   **`spots`:** List of dictionaries defining stellar spots (used only if `scenario='spots'`).
*   **`animation_settings`:** Output filename structure, FPS, number of frames, DPI.
*   **`sonification_settings`:** Audio sample rate, frequency mapping, amplitude.
*   **`ffmpeg_settings`:** Video/audio bitrates and codecs for merging.
*   **`visual_settings`:** Dark mode color theme overrides.

See comments within `config.yaml` for details on each parameter.

## Usage

Run the script from your terminal within the activated virtual environment.

**Basic Usage:**

```bash
# Generate Limb Darkening scenario using defaults from config.yaml
python transit_anim_config.py ld

# Generate Spots scenario
python transit_anim_config.py spots

# Generate No Limb Darkening scenario
python transit_anim_config.py none
```
*Output video will be named according to `output_basename` in `config.yaml` (e.g., `transit_animation_ld_dark_sound.mp4`).*

**Command-Line Options:**

```bash
# Use a specific configuration file
python transit_anim_config.py ld --config my_settings.yaml

# Override the output filename
python transit_anim_config.py spots --outfile custom_transit.mp4

# Disable sound generation for this run (overrides config/library availability)
python transit_anim_config.py ld --nosound

# Combine options
python transit_anim_config.py spots --config big_planet.yaml --outfile big_planet_transit.mp4 --nosound
```

## Output

The script generates an `.mp4` video file in the directory where you run the command. If sound is enabled and FFmpeg is working, the audio track will be embedded in the video.

## Troubleshooting

*   **`FileNotFoundError: [Errno 2] No such file or directory: 'ffmpeg'`:** FFmpeg is not installed or not in your system's PATH. Verify installation.
*   **`soundfile.LibsndfileNotFoundError`:** The underlying `libsndfile` library required by `soundfile` might be missing. Installation methods vary by OS (e.g., `brew install libsndfile` on macOS, `sudo apt-get install libsndfile1` on Debian/Ubuntu).
*   **Video saves but has no sound:** Check the script's console output for FFmpeg errors during the merging step. Ensure FFmpeg is correctly installed. Check if the temporary `.wav` file was created.
*   **`ModuleNotFoundError`:** Make sure you have activated your virtual environment and run `pip install -r requirements.txt`.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
