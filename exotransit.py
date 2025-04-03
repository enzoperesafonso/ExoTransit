# -*- coding: utf-8 -*-
"""
Generates an animation of a selected exoplanet transit scenario
(No LD, LD, LD+Spots) using parameters from a YAML config file,
with flux sonification EMBEDDED in the output video.
DARK MODE VERSION - With Phase Continuity Fix for Audio.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import batman
import argparse
from tqdm import tqdm
import os
import subprocess
import tempfile
import shutil
import yaml # Added for reading config file

# --- Check for and Import Soundfile ---
try:
    import soundfile as sf
    SOUND_SUPPORT_AVAILABLE = True # Indicates library is installed
    print("Soundfile library found, sonification can be generated.")
except ImportError:
    print("Warning: 'soundfile' library not found. Cannot generate audio for video.")
    print("Install it using: pip install soundfile")
    SOUND_SUPPORT_AVAILABLE = False
except Exception as e:
    print(f"Warning: Error importing soundfile ({e}). Audio generation disabled.")
    SOUND_SUPPORT_AVAILABLE = False

# Default visual settings (used if not specified in config)
DEFAULT_VISUAL_SETTINGS = {
    'dark_colormap': 'inferno',
    'line_color': 'cyan',
    'marker_color': 'yellow',
    'text_color': 'white',
    'grid_color': 'grey',
    'fig_bg_color': '#1c1c1c',
    'planet_color': 'black'
}

# --- Function to calculate Circle Overlap Area ---
# (Same as before)
def circle_overlap_area(x1, y1, r1, x2, y2, r2):
    d_sq = (x1 - x2)**2 + (y1 - y2)**2; d = np.sqrt(d_sq)
    if d >= r1 + r2: return 0.0
    if d <= abs(r1 - r2): return np.pi * min(r1, r2)**2
    if d < 1e-10: return np.pi * min(r1, r2)**2
    term1 = (d**2 - r2**2 + r1**2) / (2 * d * r1) if r1 > 1e-9 else 0
    term2 = (d**2 + r2**2 - r1**2) / (2 * d * r2) if r2 > 1e-9 else 0
    term1 = np.clip(term1, -1.0, 1.0); term2 = np.clip(term2, -1.0, 1.0)
    try: angle1 = 2 * np.arccos(term1)
    except ValueError: angle1 = 0.0 if term1 > 0 else 2*np.pi
    try: angle2 = 2 * np.arccos(term2)
    except ValueError: angle2 = 0.0 if term2 > 0 else 2*np.pi
    area = 0.5 * r1**2 * (angle1 - np.sin(angle1)) + 0.5 * r2**2 * (angle2 - np.sin(angle2))
    return area

# --- Function to get projected spot position ---
# (Same as before)
def get_projected_spot_coords(time_days, spot, P_rot_days, R_star):
    lon_rad = np.radians(spot['lon']); lat_rad = np.radians(spot['lat'])
    omega_rot = 2 * np.pi / P_rot_days
    current_lon_rad = lon_rad - omega_rot * time_days
    is_visible_hemi = np.abs(np.mod(current_lon_rad + np.pi, 2 * np.pi) - np.pi) < (np.pi / 2.0)
    if not is_visible_hemi: return 0, 0, False
    x = R_star * np.cos(lat_rad) * np.sin(current_lon_rad)
    y = R_star * np.sin(lat_rad)
    if x**2 + y**2 >= R_star**2: return x, y, False
    return x, y, True

# --- Function to Generate Full Audio Track ---
def generate_audio_track(flux_values, config, num_frames, total_duration_s, min_flux_val, max_flux_val, sound_enabled_for_run):
    """
    Generates a complete audio waveform using parameters from config,
    maintaining phase continuity between frames to reduce jitter/clicks.
    """
    if not sound_enabled_for_run: # Check the flag passed for this specific run
        return None

    print("Generating full audio track (with phase continuity)...")
    # Get sound params from config
    snd_cfg = config['sonification_settings']
    snd_sample_rate = int(snd_cfg.get('sample_rate', 44100))
    snd_base_freq = float(snd_cfg.get('base_freq', 220.0))
    snd_max_freq_shift = float(snd_cfg.get('max_freq_shift', 660.0))
    snd_amplitude = float(snd_cfg.get('amplitude', 0.3))
    video_fps = int(config['animation_settings'].get('video_fps', 30))

    samples_per_frame = int(snd_sample_rate / video_fps)
    total_samples = samples_per_frame * num_frames
    audio_waveform = np.zeros(total_samples)

    flux_range = max_flux_val - min_flux_val
    if flux_range < 1e-6: flux_range = 1e-6
    min_freq = snd_base_freq
    max_freq = snd_base_freq + snd_max_freq_shift

    t_audio_frame = np.linspace(0., samples_per_frame / snd_sample_rate, samples_per_frame, endpoint=False)
    frame_duration_s = samples_per_frame / snd_sample_rate

    current_phase = 0.0
    start_sample = 0
    for i in tqdm(range(num_frames), desc="Generating Audio Frames"):
        flux_value = flux_values[i]
        normalized_flux = np.clip((flux_value - min_flux_val) / flux_range, 0.0, 1.0)
        current_freq = min_freq + normalized_flux * (max_freq - min_freq)
        frame_waveform = snd_amplitude * np.sin(2 * np.pi * current_freq * t_audio_frame + current_phase)
        end_sample = start_sample + samples_per_frame
        audio_waveform[start_sample:end_sample] = frame_waveform
        start_sample = end_sample
        phase_change = 2 * np.pi * current_freq * frame_duration_s
        current_phase = (current_phase + phase_change) % (2 * np.pi)

    print("Audio track generated.")
    return audio_waveform


# --- Main Animation Function ---
def generate_transit_animation(scenario, config, sound_enabled_for_run, output_override=None):
    """
    Generates a dark mode animation using parameters from the config dictionary.

    Args:
        scenario (str): 'none', 'ld', or 'spots'.
        config (dict): Loaded configuration dictionary.
        sound_enabled_for_run (bool): Whether to generate sound for this run.
        output_override (str, optional): Path to override the output file from config.
    """
    print(f"--- Generating animation for scenario: {scenario} ---")

    # Get settings from config, providing defaults where needed
    sys_cfg = config.get('system_params', {})
    star_cfg = config.get('star_params', {})
    spots_cfg = config.get('spots', []) # Defaults to empty list
    anim_cfg = config.get('animation_settings', {})
    snd_cfg = config.get('sonification_settings', {}) # Needed for sample rate even if sound disabled later
    ffmpeg_cfg = config.get('ffmpeg_settings', {})
    # Get visual settings, merging with defaults
    vis_cfg = {**DEFAULT_VISUAL_SETTINGS, **config.get('visual_settings', {})}

    # Determine output filename
    output_basename = anim_cfg.get('output_basename', "transit_animation_{scenario}_dark_sound")
    final_output_file = output_override if output_override else f"{output_basename.format(scenario=scenario)}.mp4"
    print(f"Output file will be: {final_output_file}")

    # Apply visual style
    plt.style.use('dark_background')
    fig_bg_color = vis_cfg['fig_bg_color']

    # --- 1. Setup System Parameters ---
    params_base = batman.TransitParams()
    params_base.t0 = float(sys_cfg.get('t0', 0.0))
    params_base.per = float(sys_cfg.get('per', 3.5))
    params_base.rp = float(sys_cfg.get('rp', 0.05))
    params_base.a = float(sys_cfg.get('a', 8.8))
    params_base.inc = float(sys_cfg.get('inc', 88.5))
    params_base.ecc = float(sys_cfg.get('ecc', 0.0))
    params_base.w = float(sys_cfg.get('w', 90.0))
    params_base.limb_dark = sys_cfg.get('limb_dark', 'quadratic')
    params_base.u = [float(u_val) for u_val in sys_cfg.get('u', [0.4, 0.2])]

    params_noLD = batman.TransitParams()
    for key in ['t0', 'per', 'rp', 'a', 'inc', 'ecc', 'w']:
        setattr(params_noLD, key, getattr(params_base, key))
    params_noLD.limb_dark = "uniform"; params_noLD.u = []

    R_star = float(star_cfg.get('r_star', 1.0))
    P_rot_days = float(star_cfg.get('p_rot_days', 25.0))
    spots = spots_cfg # Use the list directly from config

    if scenario == 'none':
        params = params_noLD
        fig_title = 'Exoplanet Transit (No Limb Darkening)'
        lc_label = 'Flux (Uniform)'
        include_spots = False
    elif scenario == 'ld':
        params = params_base
        fig_title = 'Exoplanet Transit (Limb Darkening)'
        lc_label = 'Flux (Limb Dark.)'
        include_spots = False
    elif scenario == 'spots':
        params = params_base
        fig_title = 'Exoplanet Transit (LD + Spots)'
        lc_label = 'Flux (LD + Spots)'
        include_spots = True
        if not spots: # Add a warning if spots scenario chosen but no spots in config
             print("Warning: 'spots' scenario selected, but no spots defined in config.")
    else: # Should be caught by argparse choices, but defensive check
        raise ValueError("Invalid scenario.")

    fig_title = (f"Exoplanet Transit ({scenario.upper()}) - "
                 f"P={params.per}d, R_p/R_*={params.rp}, i={params.inc}°")

    # --- 2. Time Setup ---
    video_fps = int(anim_cfg.get('video_fps', 30))
    num_frames = int(anim_cfg.get('num_frames', 500))

    b = params.a * np.cos(np.radians(params.inc))
    try:
        T14_approx = params.per / np.pi * np.arcsin( R_star / params.a * np.sqrt((1 + params.rp)**2 - b**2) / np.sin(np.radians(params.inc)) )
    except ValueError:
        T14_approx = params.per * (R_star + params.rp * R_star) / (np.pi * params.a)
        print("Warning: Transit may be grazing/absent. Using simplified T14.")

    total_duration_hours_anim = T14_approx * 24.0 * 2.0
    print(f"Animation frames: {num_frames}, Target FPS: {video_fps}")
    anim_time_hours = np.linspace(-total_duration_hours_anim / 2, total_duration_hours_anim / 2, num_frames)
    anim_time_days = anim_time_hours / 24.0
    total_duration_s_anim = num_frames / video_fps
    print(f"Calculated video duration: {total_duration_s_anim:.2f} s")

    # --- 3. Calculate Model Light Curve ---
    print("Calculating light curve model...")
    m_model = batman.TransitModel(params, anim_time_days)
    flux_base = m_model.light_curve(params)
    flux_final = np.copy(flux_base)
    planet_r = params.rp * R_star
    omega_orbit = 2 * np.pi / params.per
    phase = omega_orbit * (anim_time_days - params.t0)
    incl_rad = np.radians(params.inc)
    planet_x_path = -params.a * np.sin(phase)
    planet_y_path = params.a * np.cos(phase) * np.cos(incl_rad)
    spot_positions_all_times = []

    if include_spots and spots: # Only calculate if needed and defined
        print("Calculating spot effects...")
        for i, t in enumerate(tqdm(anim_time_days, desc="Calculating Spot Effects")):
            planet_x, planet_y = planet_x_path[i], planet_y_path[i]
            is_transiting = (planet_x**2 + planet_y**2) < (R_star + planet_r)**2
            total_flux_correction = 0.0
            current_spots_vis = []
            for spot in spots:
                # Make sure spot dict keys are present
                spot_lon = spot.get('lon', 0)
                spot_lat = spot.get('lat', 0)
                spot_rad = spot.get('radius', 0)
                spot_con = spot.get('contrast', 1.0)
                spot_x, spot_y, is_visible = get_projected_spot_coords(t, spot, P_rot_days, R_star)
                spot_r_abs = spot_rad * R_star
                current_spots_vis.append({'x': spot_x, 'y': spot_y, 'r': spot_r_abs, 'visible': is_visible, 'contrast': spot_con})
                if is_transiting and is_visible:
                    overlap_A = circle_overlap_area(planet_x, planet_y, planet_r, spot_x, spot_y, spot_r_abs)
                    if overlap_A > 1e-11:
                        mu_spot = np.sqrt(max(0.0, 1.0 - (spot_x**2 + spot_y**2) / R_star**2))
                        # Use u from base params (LD) for spot correction calculation
                        u1 = params_base.u[0] if len(params_base.u)>0 else 0
                        u2 = params_base.u[1] if len(params_base.u)>1 else 0
                        I_spot_loc = 1.0 - u1 * (1 - mu_spot) - u2 * (1 - mu_spot)**2
                        flux_correction = (overlap_A / (np.pi * R_star**2)) * I_spot_loc * (1.0 - spot_con)
                        total_flux_correction += flux_correction
            flux_final[i] += total_flux_correction
            spot_positions_all_times.append(current_spots_vis)
    else: # Fill with dummies if no spots needed/defined
        for _ in anim_time_days: spot_positions_all_times.append([])

    flux_final_ppt = (flux_final - 1) * 1000
    flux_min_for_sound = flux_final.min()
    flux_max_for_sound = flux_final.max()

    # --- 4. Generate Audio Track ---
    audio_data = None
    temp_audio_file = None
    # Check local flag AND if sound support is available
    if sound_enabled_for_run and SOUND_SUPPORT_AVAILABLE:
        audio_data = generate_audio_track(
            flux_final, config, num_frames, total_duration_s_anim,
            flux_min_for_sound, flux_max_for_sound, sound_enabled_for_run
        )
        if audio_data is not None:
            temp_audio_fd, temp_audio_file = tempfile.mkstemp(suffix='.wav')
            os.close(temp_audio_fd)
            print(f"Saving temporary audio to: {temp_audio_file}")
            try:
                snd_sample_rate = int(snd_cfg.get('sample_rate', 44100))
                sf.write(temp_audio_file, audio_data, snd_sample_rate)
                print("Temporary audio saved successfully.")
            except Exception as e:
                print(f"Error saving temporary audio file: {e}")
                sound_enabled_for_run = False # Disable sound for *this run* if saving fails
                temp_audio_file_path = temp_audio_file
                temp_audio_file = None
                if temp_audio_file_path and os.path.exists(temp_audio_file_path):
                     try: os.remove(temp_audio_file_path)
                     except OSError as rm_e: print(f"Could not remove failed temp audio {temp_audio_file_path}: {rm_e}")
    else:
         # Ensure flag reflects reality if library wasn't found
         sound_enabled_for_run = False


    # --- 5. Setup Plot for Animation ---
    fig, (ax_star, ax_lc) = plt.subplots(1, 2, figsize=(12, 5.5),
                                         gridspec_kw={'width_ratios': [1, 1.2]})
    fig.set_facecolor(fig_bg_color)
    sound_status_msg = "(Sound in File)" if sound_enabled_for_run and temp_audio_file else "(No Sound in File)"
    fig.suptitle(f'{fig_title} {sound_status_msg}',
                 fontsize=14, color=vis_cfg['text_color'])
    fig.tight_layout(pad=3.0, rect=[0, 0.03, 1, 0.93])

    # Star plot setup using vis_cfg
    ax_star.set_aspect('equal'); ax_star.set_title("Transit View", color=vis_cfg['text_color'])
    ax_star.set_xlabel("X [$R_*$]", color=vis_cfg['text_color']); ax_star.set_ylabel("Y [$R_*$]", color=vis_cfg['text_color'])
    ax_star.set_xlim(-R_star * 1.3, R_star * 1.3); ax_star.set_ylim(-R_star * 1.3, R_star * 1.3)
    ax_star.tick_params(axis='x', colors=vis_cfg['text_color']); ax_star.tick_params(axis='y', colors=vis_cfg['text_color'])
    for spine in ax_star.spines.values(): spine.set_color(vis_cfg['text_color'])
    star_resolution=100; x_grid=np.linspace(-R_star,R_star,star_resolution); y_grid=np.linspace(-R_star,R_star,star_resolution)
    X_grid,Y_grid=np.meshgrid(x_grid,y_grid); R_grid=np.sqrt(X_grid**2+Y_grid**2)
    if params.limb_dark == "uniform": intensity=np.ones_like(R_grid); vmin_star,vmax_star=0.0,1.5
    else: mu=np.sqrt(np.maximum(0,1-(R_grid/R_star)**2)); u1=params.u[0]; u2=params.u[1] if len(params.u)>1 else 0; intensity=1-u1*(1-mu)-u2*(1-mu)**2; vmin_star,vmax_star=0.0,1.0
    intensity[R_grid>R_star]=np.nan
    ax_star.imshow(intensity,extent=[-R_star,R_star,-R_star,R_star],cmap=vis_cfg['dark_colormap'],origin='lower',vmin=vmin_star,vmax=vmax_star,zorder=1)
    planet_patch=patches.Circle((planet_x_path[0],planet_y_path[0]),planet_r,color=vis_cfg['planet_color'],zorder=10); ax_star.add_patch(planet_patch); planet_patch.set_visible(False)
    spot_patches=[]
    spot_contrast_cmap = plt.get_cmap(vis_cfg['dark_colormap'])
    if include_spots and spots:
        for spot_info in spots:
            # Get radius/contrast safely
            spot_rad = spot_info.get('radius', 0)
            spot_con = spot_info.get('contrast', 1.0)
            patch_color=spot_contrast_cmap(spot_con*0.6); patch=patches.Circle((0,0),spot_rad*R_star,color=patch_color,zorder=5); ax_star.add_patch(patch); patch.set_visible(False); spot_patches.append(patch)

    # Light curve plot setup using vis_cfg
    ax_lc.set_title("Light Curve", color=vis_cfg['text_color']); ax_lc.set_xlabel("Time from Midpoint [h]", color=vis_cfg['text_color']); ax_lc.set_ylabel("Relative Flux [ppt]", color=vis_cfg['text_color'])
    ax_lc.set_xlim(anim_time_hours.min(), anim_time_hours.max()); min_flux_lim=flux_final_ppt.min(); max_flux_lim=flux_final_ppt.max()
    ax_lc.set_ylim(min_flux_lim*1.1-5,max(5,max_flux_lim)*1.1+5); ax_lc.grid(True,linestyle=':',alpha=0.4,color=vis_cfg['grid_color'])
    ax_lc.tick_params(axis='x', colors=vis_cfg['text_color']); ax_lc.tick_params(axis='y', colors=vis_cfg['text_color'])
    for spine in ax_lc.spines.values(): spine.set_color(vis_cfg['text_color'])
    line_lc,=ax_lc.plot([],[],color=vis_cfg['line_color'],linestyle='-',lw=1.5,label=lc_label)
    marker_lc,=ax_lc.plot([],[],marker='o',color=vis_cfg['marker_color'],markersize=6,linestyle='None')
    leg=ax_lc.legend(loc='lower right',fontsize='small'); plt.setp(leg.get_texts(),color=vis_cfg['text_color'])


    # --- 6. Animation Functions (init & update - NO SOUND HERE) ---
    artists_to_update = [planet_patch] + spot_patches + [line_lc, marker_lc]
    def init():
        planet_patch.center = (planet_x_path[0], planet_y_path[0])
        planet_patch.set_visible(False)
        for patch in spot_patches: patch.set_visible(False)
        line_lc.set_data([], [])
        marker_lc.set_data([], [])
        return artists_to_update
    def update(i):
        # ... (Visual update logic remains the same) ...
        current_anim_time_h=anim_time_hours[i]; current_flux_ppt=flux_final_ppt[i]
        planet_x=planet_x_path[i]; planet_y=planet_y_path[i]
        planet_patch.center=(planet_x,planet_y); is_overlapping=np.sqrt(planet_x**2+planet_y**2)<(R_star+planet_r); planet_patch.set_visible(is_overlapping)
        if include_spots and spots:
            current_spots_vis=spot_positions_all_times[i]
            for patch, spot_vis in zip(spot_patches, current_spots_vis):
                patch.set_visible(spot_vis['visible'])
                if spot_vis['visible']: patch.center=(spot_vis['x'],spot_vis['y'])
        line_lc.set_data(anim_time_hours[:i+1],flux_final_ppt[:i+1]); marker_lc.set_data([current_anim_time_h],[current_flux_ppt])
        return artists_to_update

    # --- 7. Create Animation Object ---
    anim_interval = 1000 / video_fps
    ani = animation.FuncAnimation(fig, update, frames=num_frames,
                                  init_func=init, blit=True, interval=anim_interval)

    # --- 8. Save Video Frames (Temporarily) and Merge with Audio ---
    temp_video_file = None
    success = False
    try:
        temp_video_fd, temp_video_file = tempfile.mkstemp(suffix='.mp4')
        os.close(temp_video_fd)

        print(f"\nSaving video frames temporarily to: {temp_video_file}...")
        ffmpeg_bitrate = int(ffmpeg_cfg.get('video_bitrate', 3000))
        dpi = int(anim_cfg.get('dpi', 200))
        writer_instance = animation.FFMpegWriter(fps=video_fps, metadata=dict(artist='Me - Configured'), bitrate=ffmpeg_bitrate)
        progress_callback = lambda current_frame, total_frames: pbar.update(1)
        with tqdm(total=num_frames, desc="Rendering Video Frames") as pbar:
            ani.save(temp_video_file, writer=writer_instance, progress_callback=progress_callback, dpi=dpi)
        print("Temporary video frames saved.")

        # --- Merge audio and video using FFmpeg ---
        if sound_enabled_for_run and temp_audio_file and os.path.exists(temp_audio_file) and os.path.exists(temp_video_file):
            print(f"Merging video and audio into final file: {final_output_file}")
            audio_codec = ffmpeg_cfg.get('audio_codec', 'aac')
            audio_bitrate = ffmpeg_cfg.get('audio_bitrate', '192k')
            ffmpeg_cmd = [
                'ffmpeg', '-y',
                '-i', temp_video_file,
                '-i', temp_audio_file,
                '-c:v', 'copy',
                '-c:a', audio_codec, '-b:a', audio_bitrate,
                '-shortest',
                final_output_file
            ]
            print(f"Executing: {' '.join(ffmpeg_cmd)}")
            try:
                result = subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True, encoding='utf-8')
                print("Video and audio merged successfully.")
                success = True
            except subprocess.CalledProcessError as e:
                # ... (Error printing same as before) ...
                print("\n--- FFmpeg Error ---"); print(f"Command failed: {' '.join(e.cmd)}"); print(f"Return code: {e.returncode}")
                stdout = e.stdout.decode('utf-8', errors='replace') if isinstance(e.stdout, bytes) else e.stdout
                stderr = e.stderr.decode('utf-8', errors='replace') if isinstance(e.stderr, bytes) else e.stderr
                print("FFmpeg stdout:", stdout); print("FFmpeg stderr:", stderr); print("--------------------\n")
                success = False
            except FileNotFoundError:
                 print("\n--- FFmpeg Error ---"); print("FFmpeg command not found..."); print("--------------------\n")
                 success = False

        elif os.path.exists(temp_video_file):
             print("Sound disabled or audio/merge failed. Saving video-only file.")
             try:
                 shutil.copyfile(temp_video_file, final_output_file)
                 print(f"Video-only file saved to {final_output_file}")
                 success = True
             except Exception as e:
                 print(f"Error copying temporary video file: {e}")
                 success = False
        else:
             print("Error: Temporary video file was not created.")
             success = False

    except Exception as e:
        print(f"\nAn error occurred during animation saving/merging: {e}")
        if isinstance(e, animation.AnimationException): print("Check FFmpeg setup/permissions.")
        success = False
    finally:
        # --- 9. Cleanup Temporary Files ---
        # ... (Cleanup logic same as before) ...
        print("Cleaning up temporary files...")
        if temp_video_file and os.path.exists(temp_video_file):
            try: os.remove(temp_video_file); # print(f"Removed temp video: {temp_video_file}")
            except OSError as e: print(f"Error removing temp video {temp_video_file}: {e}")
        if temp_audio_file and os.path.exists(temp_audio_file):
            try: os.remove(temp_audio_file); # print(f"Removed temp audio: {temp_audio_file}")
            except OSError as e: print(f"Error removing temp audio {temp_audio_file}: {e}")
        plt.close(fig)
        print("Cleanup finished.")
        if success: print(f"\nFinal animation {'with sound ' if sound_enabled_for_run and temp_audio_file else ''}saved successfully to: {final_output_file}")
        else: print(f"\nFailed to create final animation file: {final_output_file}")


# --- Command Line Argument Parsing ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a DARK MODE transit animation using a config file, with optional SOUND.")
    parser.add_argument(
        'scenario',
        type=str,
        choices=['none', 'ld', 'spots'],
        help="Select the transit scenario: 'none' (No LD), 'ld' (LD spotless), 'spots' (LD + Spots)."
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml', # Default config file name
        help="Path to the YAML configuration file (default: config.yaml)."
    )
    parser.add_argument(
        '--outfile',
        type=str,
        default=None,
        help="Optional path to override the output video file specified in the config."
    )
    parser.add_argument(
        '--nosound',
        action='store_true',
        help="Disable audio generation, overriding config/library availability."
    )

    args = parser.parse_args()

    # --- Load Configuration ---
    config_data = {}
    try:
        with open(args.config, 'r') as f:
            config_data = yaml.safe_load(f)
        if not isinstance(config_data, dict): # Basic check if loading failed or format is wrong
             print(f"Error: Config file '{args.config}' is not a valid YAML dictionary.")
             config_data = {} # Use empty dict to proceed with defaults/errors later
        else:
             print(f"Loaded configuration from '{args.config}'")
    except FileNotFoundError:
        print(f"Error: Config file '{args.config}' not found. Using defaults where possible.")
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file '{args.config}': {e}")
        print("Proceeding with defaults where possible.")

    # Determine if sound should be enabled for this run
    # Start with library availability, then check the --nosound flag
    sound_enabled_this_run = SOUND_SUPPORT_AVAILABLE
    if args.nosound:
        print("Audio generation explicitly disabled via --nosound flag.")
        sound_enabled_this_run = False # Override

    # Check for ffmpeg availability early
    try:
        is_windows = os.name == 'nt'
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True, text=True, shell=is_windows, encoding='utf-8')
        print("FFmpeg found.")
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("\n*** WARNING: FFmpeg command not found or failed. ***")
        if sound_enabled_this_run:
             print("Audio will not be merged into the video. Ensure FFmpeg is installed and in PATH for sound output.")
             # Optionally disable sound here if ffmpeg is required:
             # sound_enabled_this_run = False
             # print("Disabling sound generation as FFmpeg is required but not found.")

    # --- Run Main Function ---
    generate_transit_animation(
        scenario=args.scenario,
        config=config_data,
        sound_enabled_for_run=sound_enabled_this_run,
        output_override=args.outfile
    )