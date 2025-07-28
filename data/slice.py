from pathlib import Path
import shutil
import pickle
import librosa as lr
import numpy as np
import soundfile as sf
from tqdm import tqdm
from pytorch3d.transforms import matrix_to_axis_angle

def slice_audio(audio_file, stride, length, out_dir):
    audio, sr = lr.load(audio_file, sr=None)
    file_name = Path(audio_file).stem
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    start_idx, idx = 0, 0
    window, stride_step = int(length * sr), int(stride * sr)

    while start_idx <= len(audio) - window:
        audio_slice = audio[start_idx : start_idx + window]
        sf.write(out_dir / f"{file_name}_slice{idx}.wav", audio_slice, sr)
        start_idx += stride_step
        idx += 1

    return idx

def slice_motion(motion_file, stride, length, num_slices, out_dir):
    with open(motion_file, "rb") as f:
        motion = np.load(f, allow_pickle=True).item()

    file_name = Path(motion_file).stem
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    pos = motion["pred_trans"].squeeze(1)
    q = matrix_to_axis_angle(motion["pred_rotmat"])
    # pos, q = motion['pos'] * motion['scale'], motion['q']
    window, stride_step = int(length * 30), int(stride * 30)
    slice_count = 0
    start_idx = 0

    while start_idx <= len(pos) - window and slice_count < num_slices:
        pos_slice, q_slice = pos[start_idx : start_idx + window], q[start_idx : start_idx + window]
        out_data = {"pos": pos_slice, "q": q_slice}
        with open(out_dir / f"{file_name}_slice{slice_count}.npy", "wb") as f:
            np.save(f, out_data, allow_pickle=True)
        start_idx += stride_step
        slice_count += 1

    return slice_count

def get_filenames(directory, extension):
    return [f.stem for f in directory.glob(f'*{extension}')]

def remove_extra_files(source_dir, target_filenames, extension):
    for file in source_dir.glob(f'*{extension}'):
        if file.stem.replace("_original", "").replace("_simplified", "") not in map(lambda x: x.replace("_original", "").replace("_simplified", ""), target_filenames):
            file.unlink()  

# Lower slice length
def slice_and_clean(motion_dir, wav_dir, stride=0.5, length=5):
    """Slice motion and audio files and remove unmatched files."""
    motion_dir, wav_dir = Path(motion_dir), Path(wav_dir)
    motion_out, wav_out = motion_dir.with_name(f"{motion_dir.name}_sliced"), wav_dir.with_name(f"{wav_dir.name}_sliced")
    motion_out.mkdir(parents=True, exist_ok=True)
    wav_out.mkdir(parents=True, exist_ok=True)

    original_motion_files = sorted(motion_dir.glob("*_original.npy"))
    simplified_motion_files = sorted(motion_dir.glob("*_simplified.npy"))
    wav_files = sorted(wav_dir.glob("*.wav"))

    assert len(wav_files) == len(original_motion_files) == len(simplified_motion_files), f"Number of audio and (original, simplified) motion file \
        pairs do not match: {len(wav_files)}, ({len(original_motion_files)}, {len(simplified_motion_files)})"

    for original_motion_file, simplified_motion_file, wav_file in tqdm(zip(original_motion_files, simplified_motion_files, wav_files), total=len(wav_files)):
        print(original_motion_file.stem, wav_file.stem)
        # assert motion_file.stem == wav_file.stem, "Mismatched file names."
        audio_slices = slice_audio(wav_file, stride, length, wav_out)

        with open(original_motion_file, "rb") as og_m:
            with open(simplified_motion_file, "rb") as spl_m:
                og_motion = np.load(og_m, allow_pickle=True).item()
                spl_motion = np.load(spl_m, allow_pickle=True).item()
                og_pos = og_motion["pred_trans"].squeeze(1)
                spl_pos = spl_motion["pred_trans"].squeeze(1)

                if len(og_pos) < len(spl_pos): 
                    # Original motion shorter than simplified motion
                    original_motion_slices = slice_motion(original_motion_file, stride, length, audio_slices, motion_out)
                    simplified_motion_slices = slice_motion(simplified_motion_file, stride, length, original_motion_slices, motion_out)
                
                else:
                    simplified_motion_slices = slice_motion(simplified_motion_file, stride, length, audio_slices, motion_out)
                    original_motion_slices = slice_motion(original_motion_file, stride, length, simplified_motion_slices, motion_out)


        assert original_motion_slices == simplified_motion_slices, f"Mismatch in slice counts: {len(original_motion_slices)}, {len(simplified_motion_slices)}"
        assert audio_slices >= original_motion_slices, "Mismatch in slice counts."

    # Clean up extra files
    motion_filenames = get_filenames(motion_out, '.npy')
    wav_filenames = get_filenames(wav_out, '.wav')
    remove_extra_files(wav_out, motion_filenames, '.wav')
    remove_extra_files(motion_out, wav_filenames, '.npy')

    motion_files = sorted(motion_dir.glob("*.npy"))
    wav_files = sorted(wav_dir.glob("*.wav"))

    assert len(wav_files) == len(motion_files) / 2
    print("motion files, wav files: ", len(motion_files), len(wav_files))



