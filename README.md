

# StyleMatte: Adversarially-Guided Portrait Matting


[[`arXiv`](https://arxiv.org/abs/2305.02981)] [[`BibTeX`](#CitingStyleMatte)] [[`Gitlab`](https://gitlab.aicloud.sbercloud.ru/rndcv/stylematte)] [[`Demo`](https://huggingface.co/spaces/befozg/stylematte)]


<p align="left">
  <img src="assets/swap.gif" width="70%" height="60%" />
</p><br/>

## Model Zoo and Baselines


| State file                           | Size     | Where to place                                   | Download |
|-----------------------------------|----------|-------------------------------------------|----|
| stylematte_pure.pth    | 133.6 MB  |  `stylematte/checkpoints/`        | [stylematte.zip](https://sc.link/lYkM5) |
| stylematte_synth.pth | 133.6 MB  |   `stylematte/checkpoints/` | [stylematte.zip](https://sc.link/lYkM5) |
| animals.pkl    | 300.5 MB  |    `stylegan3`      | [stylemattegan.zip](https://sc.link/kRjO5) |
| humans.pkl    | 281.1 MB  |    `stylegan3`      | [stylemattegan.zip](https://sc.link/kRjO5) |



## How to run StyleMatteGAN
To synthesize synthetic dataset of RGBA images, move to `stylegan3` and run `synthesize.py`. You should create conda environment from `stylegan3/environment.yml`. You can also generate images with different truncation and seed values using `gen_images.py` .

```bash
conda activate stylegan3
cd stylegan3
python synthesize.py
```

To change image background on synthetic dataset, run 
```bash
python visualizer.py
```

In the GUI you can choose model weights, background picture, truncation value and other visualization parameters. 

## StyleMatteGAN results

<p align="left">
  <img src="assets/back.gif" width="60%" height="60%" />
</p><br/>

## How to run StyleMatte
To test our model, change directory to `stylematte` and run `test.py`. You can modify `test.yaml` file for your datasets and models.

```bash
cd stylematte
python test.py
```

The report directory is `stylematte/report/`. See report examples there.

## StyleMatte results

<p align="left">
  <img src="assets/merged.gif" width="100%" height="50%" />
</p><br/>

### License
<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br />This work is licensed under a variant of <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.

Please see the specific [license](https://github.com/chroneus/stylematte/tree/main/license/en_us.pdf).
