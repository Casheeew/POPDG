# SimpliFie

## Build Instructions

1. Clone this repository and setup the environment according to [POPDG.](https://github.com/Luke-Luo1/POPDG)

```bash
git clone https://github.com/Casheeew/POPDG
```

2. Download the motions and audio files (`simplifie_dataset`) from [Google Drive](https://drive.google.com/drive/folders/1Uoa1bgp01yxhVXLnITwBWm08Mt6zbYzd) and put it in `data/`. Alternatively, you can build the dataset yourself by referring to the [appendix](#manually-build-the-dataset).

3. Create the dataset.

```bash
cd data
python create_dataset.py --dataset_folder simplifie_dataset --extract-baseline --extract-jukebox
```
By default, this processes the data into slices of length 5 with stride 0.5. If you wish to change this, you can modify the parameters in `create_dataset.py`.

### Pre-trained checkpoints

You can find our pre-trained checkpoints here:



## Train the model
1. Download the pre-trained `checkpoint.pt` from [POPDG](https://github.com/Luke-Luo1/POPDG) and put it at the top-level directory.

2. Train the model:

```bash
python train.py --batch_size 50 --epochs 2000 --exp_name <experiment name> --feature_type jukebox
 --checkpoint checkpoint.pt --save_interval 5                       
```

If you use `accelerate` to train the model with multiple GPUs, you can also use the following command:

```bash
accelerate launch train.py --batch_size <batch_size> --epochs 2000 --exp_name <experiment name> --feature_type jukebox --checkpoint checkpoint.pt --save_interval 5
```
Other flags and their usage can be found in `args.py`.

> **_NOTE:_**  Whenever you make changes to the dataset or to `load_popdanceset.py`, be sure to run the commands with the --force_reload and --no_cache flags. Otherwise, the model will use the previously cached dataset instead of your updates.

You can experiment with `guidance_weight` to see how it affects the simplification process.

## Model Evaluation

1. If you want to create evaluation data from the built test data, run the following:

```bash
python create_eval_data.py
```

To evaluate custom or in-the-wild data, add .npy motion files (in TRAM output format) and .wav music audio files in pairs (with the same name).

2. Test the model's dance generation:

```bash
python test.py --checkpoint <model checkpoint> --music_dir eval_data --render_dir test_renders --save_motions --out_length=5 --motion_save_dir eval/motions --no_render
```

If you wish to see the generated results, remove the `--no_render` option.

3. Evaluate the generated results

Below are the commands to test PFC, PBC, DIV, FID and dance complexity scores, respectively.

```bash
python eval/eval_pfc.py --motion_path eval/motions
python eval/eval_pbc.py --motion_path eval/motions
python eval/calculate_scores.py --motion_path eval/motions
python eval/calculate_simscore.py --motion_path eval/motions
```


## Appendix

### Manually build the dataset

If you wish to build the dataset from scratch, please follow the steps below:

1. Run the videos through [TRAM](https://github.com/yufu-wang/tram) to obtain the motion data:

- Setup the TRAM model via to the provided instructions. 

- Obtain the annotated and segmented data from [Google Drive](https://drive.google.com/drive/folders/1Uoa1bgp01yxhVXLnITwBWm08Mt6zbYzd). Put the `annotated_and_segmented` folder at the root directory of the TRAM repository.

- Create the file `run_all.sh` at the root directory of the TRAM repository and copy the following:

```bash
#!/bin/bash

VIDEO_DIR="./annotated_and_segmented"
EXT="mp4"  # or "mp4" depending on your files

for strategy in "1" "2" "3" "4" "5" "6"; do
    for video in "$VIDEO_DIR"/$strategy/*.$EXT; do
        echo "▶️ Processing: $video"
        
        # Step 1: Estimate Camera
        python scripts/estimate_camera.py --video "$video" --strategy $strategy --static_camera || { echo "❌ Failed camera: $video"; continue; }

        # Step 2: Estimate Humans
        python scripts/estimate_humans.py --video "$video" --strategy $strategy || { echo "❌ Failed humans: $video"; continue; }

        echo "✅ Done: $video"
        echo
    done
done

echo "🏁 All done!"
```

Running this in a single process will take around two days, and it's recommended to split this into multiple processes.

- Run the model:

```bash
bash run_all.sh
```

2. Extract .wav audio files

You can refer to the script below:

```python
import os
from os import path
from tqdm import tqdm

if __name__ == "__main__":    
    data_dir = "annotated_and_segmented"
    strats = sorted(os.listdir(data_dir))

    for strat in strats:
        print(f"Processing strategy: {strat}\n")
        if strat == 'analysis_results_english.csv':
            continue

        files = sorted(os.listdir(path.join(data_dir, strat)))
        for f in tqdm(files):
            print(f"Processing file: {f}")
            out_path = f.replace(".mp4", ".wav")
            in_path = path.join(".", data_dir, strat, f)
            
            o1 = os.system(
                f"ffmpeg -loglevel error -y -i {in_path} wav/{strat}/{out_path}"
            )
```

3. Assemble the dataset
Motion data are located at `hps_track_0.npy` in TRAM's `results/` directory. Combine motion files with their .wav audio files to create the following structure:

```bash
simplifie_dataset/
├── [1-6]/
│ ├── motions/ # Motion data files
│ └── wavs/ # Corresponding audio files
```