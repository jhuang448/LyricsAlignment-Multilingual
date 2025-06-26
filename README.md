# LyricsAlignment-Multilingual

This repository is a multilingual adaptation of the following work on English lyrics alignment:

Jiawen Huang, Emmanouil Benetos, Sebastian Ewert, "**Improving Lyrics Alignment through Joint Pitch Detection**," 
International Conference on Acoustics, Speech and Signal Processing (ICASSP). 2022. [https://ieeexplore.ieee.org/document/9746460](https://ieeexplore.ieee.org/document/9746460)

The old repository is: [https://github.com/jhuang448/LyricsAlignment-MTL](https://github.com/jhuang448/LyricsAlignment-MTL).

## Dependencies

This repo is written in python 3.9. Pytorch is used as the deep learning framework. To install the required python package, run the following (We have updated the environment from the old repository. If things do not work, please check the environment.)

```
pip install -r requirements.txt
```

Install phonemizer: [https://github.com/bootphon/phonemizer](https://github.com/bootphon/phonemizer)

Besides, you might want to install some source-separation tool (e.g. [Spleeter](https://github.com/deezer/Spleeter), [Open-Unmix](https://github.com/sigsep/open-unmix-pytorch))
or use your own system to prepare source-separated vocals.

## Run Inference on a Jamendo example

Phonemizer and source-separation tool are not required to run this example:

```
python eval.py --sepa_dir=./jamendo_example/audio/ --annot_dir=./jamendo_example/annot/ --load_model=./checkpoints/checkpoint_Baseline --pred_dir=./jamendo_example/pred/ --model=baseline --cuda --ext .mp3
```

The generated csv files under `pred_dir`.

## Prepare Training and Evaluation Data

The **DALI v2.0** is required for training. See instructions on how to get the dataset: [https://github.com/gabolsgabs/DALI](https://github.com/gabolsgabs/DALI). 

The annotated **Multi-Lang Jamendo** is used for evaluation: [https://huggingface.co/datasets/jamendolyrics/jamendolyrics](https://huggingface.co/datasets/jamendolyrics/jamendolyrics)

All the songs in both datasets need to be separated and saved in advance. 
Training split and the phonemized lyrics annotations for evaluation can be downloaded [here](https://drive.google.com/drive/folders/1upoZQjBwpKx5-zge9DpeedJ04H8K4I5F?usp=sharing) and placed at the repository root.

## Training

```
python train.py --sepa_dir=/path/to/separated/DALI/vocals/ 
                --checkpoint_dir=/where/to/save/checkpoints/ --log_dir=/where/to/save/tensorboard/logs/ 
                --model=baseline --cuda
```

Run `python train.py -h` for more options.

If the full training split (_multilingual_split.npy_ from the drive link above) is not downloaded, it will run a debug session with _dummy_split.npy_.

## Cite this work

```
@inproceedings{jhuang_icassp2022,
  author       = {Jiawen Huang and
                  Emmanouil Benetos and
                  Sebastian Ewert},
  title        = {Improving Lyrics Alignment Through Joint Pitch Detection},
  booktitle    = {{IEEE} International Conference on Acoustics, Speech and Signal Processing,
                  {ICASSP} 2022, Virtual and Singapore, 23-27 May 2022},
  pages        = {451--455},
  publisher    = {{IEEE}},
  year         = {2022}
}
```


## Contact

Jiawen Huang

jiawen.huang@qmul.ac.uk
