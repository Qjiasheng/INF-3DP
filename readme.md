# INF-3DP: Implicit Neural Fields for Collision-Free Multi-Axis 3D Printing

<p align="center">
  <a href="https://qjiasheng.github.io/crml/inf3dp/">
    <img src="https://img.shields.io/badge/Project%20Page-222222?logo=google&logoColor=skyblue" alt="Project Page">
  </a>
  <a href="https://arxiv.org/pdf/2509.05345">
    <img src="https://img.shields.io/badge/arXiv-b31b1b?logo=arxiv&logoColor=white" alt="arXiv">
  </a>
  <a href="https://youtu.be/uhsC-nGu0lw">
    <img src="https://img.shields.io/badge/YouTube-%23FF0000.svg?style=Flat&logo=YouTube&logoColor=white" alt="YouTube Video">
  </a>
  <a href="https://www.bilibili.com/video/BV1kMH6zREVy/?vd_source=a598467ab21890aa22f71bf6590ee097">
    <img src="https://img.shields.io/badge/bilibili-00A1D6.svg?style=Flat&logo=bilibili&logoColor=white" alt="BiliBili Video">
  </a>

> SIGGRAPH Asia 2025 (ACM Transactions on Graphics)
>
> [Jiasheng Qu<sup>1</sup> ](https://qjiasheng.github.io/), Zhuo Huang<sup>1, 2</sup>, Dezhao Guo<sup>1</sup>, Hailin Sun<sup>1</sup>, Aoran Lyu<sup>2</sup>, [Chengkai Dai<sup>3</sup>](https://chengkai-dai.github.io/), Yeung Yam<sup>1,3</sup>, and [Guoxin Fang<sup>1, 3, *</sup>](https://guoxinfang.github.io/) 
>
> 1 The Chinese University of Hong Kong, China. 2 The University of Manchester, Manchester, United Kingdom. 3 Centre for Perceptual and Interactive Intelligence, Hong Kong, China.

# Overview

INF-3DP is a renewed framework for multi-axis 3D printing, using field computing with implicit neural fields (INFs) across all stages. 

Please refer to our paper and video to explore its capabilities and how it can meet your needs. The key features of INF-3DP include:

+ **Scalable** (handle million-level waypoints), and **collision-free** (differentiable motion planning with Time-Varying SDF)
+ **Support-free** printing, and superior **surface quality** (optimize singularity) 

<img src="./md_assets/teaser.png" alt="teaser" style="zoom:80%;" /> 




# Guidance

>  The `bunny` and `fertility` models are used as examples throughout this document. You can try using your own models by following this guidance.  Feel free to raise issues that you may encounter. :blush: 


- [Installation](#installation)
  - [Quick start](##quick-start)
  - [Test your models](##test-your-models)
  
- [Project Structure](#Project-Structure) 
- [Usages](#usages)
  - [Viewer](##viewer)
  - [ SDF](##SDF) 
  - [Guidance field](##guidance-field)
  - [Infill field](##infill-field)
  - [Toolpath generation](##toolpath-generation)
  - [Level field](##level-field)
  - [TV-SDF](##tv-sdf)
  - [Quaternion field and collision response](##quaternion-field-and-collision-response) 
  

+ [Others](#Others)
  + [BibTeX](##BibTeX)
  + [Acknowledgements](##Acknowledgements)

# Installation

We build and run this project on Ubuntu 22.04, Nvidia RTX 4090 (24 GB), and 64 GB memory. We use Python 3.10,  CUDA version is 12.1.

> :point_right:  Pls check cuda version first  with `nvcc --version`. if same `12.1`, directly use cmds below to set the env. If not, pls make sure torch version is compatible with your cuda version, and refactor some lines in `pyproject.toml`, by checking [Installing previous versions of PyTorch](https://pytorch.org/get-started/previous-versions/). 
>
> ```bash
> # you may refactor the versions 
> ...
> torch = "^2.3.0"
> torchvision = "^0.18.0"
> torchaudio = "^2.3.0"
> 
> [[tool.poetry.source]]
> name = "pytorch-cu121"
> url = "https://download.pytorch.org/whl/cu121"
> priority = "explicit"
> ...
> 
> # NOTE e.g. here we select cmd from webpage
> pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
> ```

+ Set  env

  ```bash
  git clone https://github.com/Qjiasheng/INF-3DP.git
  cd INF-3DP
  
  conda create --name inf python=3.10
  conda activate inf
  
  pip install poetry
  poetry install
  ```

+ [Optional] Install custom cuda package for parallel toolpath generation. 

  ```bash
  cd contour_cuda
  pip install .
  ```

## Quick start

Trained example checkpoints, waypoint files, config files that we previously tested are provided in [Google Drive](https://drive.google.com/drive/folders/1PtbKpwon7thvAkUZt09X7_fMXDs1blsy?usp=sharing). Download and place them under the project path, then run scripts to have a quick test.  Key related results can be viewed in [Usages](#usages).

:point_right:  If users encounter out-of-memory issues, please reduce the training batch size and the isovalue numbers during toolpath generation.

```bash 
# SDF test
python exp_scripts/test_sdf.py --experiment_name=bunnyz --config_path=configs/sdf_config.yaml
# first-stage slice field test
python exp_scripts/test_slice.py --experiment_name=bunny_init --config_path=configs/init_slice_config.yaml
# toolpath generation
python exp_scripts/toolpath.py --experiment_name=bunny_single_shell --config_path=configs/toolpath_config.yaml
...
# launch visualizer to check field volume. file paths need to specify in `slice_view_pcd.py`
python slice_view_pcd.py
```

:rabbit2: For all train/ test applications, make sure parameters block is well set under `experiment_name`  in corresponding  `config_path` files.

## Test your models

> Below is a brief workflow for testing your models. 
>
> Training loss/ scalar field cross sections can be checked with `tensorboard`.  Launch it if needed 
>
> ```bash
> tensorboard --logdir ./
> ```

#### 1 Prepare pcd with normals

Pcd file `.xyz` has lines containing  `coordinates, normals`  can be exported from [Meshlab](https://www.meshlab.net/). Put pcd files under `sdf_data/` and set parameters in `configs/sdf_config.yaml` similar as I offered.

#### 2 Train/ Test SDF

```bash
# train SDF, taking long time. Urs can set smaller training epochs
python exp_scripts/train_sdf.py --experiment_name=bunnyz --config_path=configs/sdf_config.yaml
```

Test SDF, also prepare training dataset (normalized pcd, curv, base_tags) for further field training.

```bash
# test when train finished
python exp_scripts/test_sdf.py --experiment_name=bunnyz --config_path=configs/sdf_config.yaml
```

#### 3 Train/ Test guidace field (two-stages)

```bash
# first stage
python exp_scripts/train_slice.py --experiment_name=bunny_init --config_path=configs/init_slice_config.yaml
# second stage
python exp_scripts/train_fs.py --experiment_name=bunny_fn --config_path=configs/fn_slice_config.yaml

# test first or second field
python exp_scripts/test_slice.py --experiment_name=bunny_init --config_path=configs/init_slice_config.yaml
python exp_scripts/test_slice.py --experiment_name=bunny_fn --config_path=configs/fn_slice_config.yaml
```

#### 4 Toolpath generation

```bash
# toolpath generation, single shell 
python exp_scripts/toolpath.py --experiment_name=bunny_single_shell --config_path=configs/toolpath_config.yaml
# infill application, NOTE must train infill field first with similar cmds
python exp_scripts/toolpath.py --experiment_name=bunny_infill --config_path=configs/toolpath_config.yaml
```

#### 5 Level/ partition 

+ similar cmds

#### 6 Collision train/ test

+ similar cmds. 

+ :point_right: We also offer `toy_tvsdf.py` to check constructed tvsdf at given working waypoints. Users can use viewer by specifying file paths to check tvsdf volume field.  In addition, the differetiable collision reponse with tvsdf to view push directions when collision happens is also provided, see example results in TV-SDF and collision reponse.

  ```bash 
  python exp_scripts/toy_tvsdf.py --experiment_name=bunny_single_shell --config_path=configs/collision_config.yaml
  
  # then launch viewer, pls specify file paths in `slice_view_pcd.py`
  # ------ tvsdf
  scale_coords = False
  sdf_volume_fn = './logs/collision/bunny_single_shell/tvsdf_volume.npz'  # tvsdf
  pcd_fn = './logs/collision/bunny_single_shell/till_waypts.xyz'
  ```

# Project structure

This project follows similar structure as [SIREN](https://github.com/vsitzmann/siren), but seperates `configs` for different feilds/ toolpath applications since there are many parameters to manage.
```yaml
/inf-3dp
|-- configs/
|   |-- XXXXX.yaml           # config parameters 
|-- contour_cuda/			 # custom cuda package for toolpath 
|-- exp_scripts/             # train & test scripts
|-- sdf_data/                # sdf training data
|-- dataio.py                # training Datasets
|-- diff_operators.py        # grad, laplace, hessian ...
|-- level_collision.py       # level coll functions
|-- loss_functions.py        # training loss funcs
|-- readme.md
|-- sdf_meshing.py           # sdf realted func
|-- slice_meshing.py         # slice field func
|-- slice_toolpath.py        # toolpath func
|-- slice_view_pcd.py        # visualizer based on pyvista and pyqt
|-- toolpath_utils.py        # specific toolpath utils
|-- toolpath_vis.py          # toolpath vis func
|-- training.py              # training loop
|-- utils.py				 # general utils func				
|-- vis.py                   # general vis func
```


# Usages

> Each field has both `train` and `test` files. Once trained, run test to check results. A viewer based on pyvista and pyqt is provided to visulize volume field.
>
> :loudspeaker: Below we show some key result examples. Users can run and refactor codes to check detailed  results.

## Viewer

+ Simple viewer to visulize volume field.
+ BBX to select points. Used to set base threshold for Neumann boundary conditions.

<img src="./md_assets/viewer.png" alt="viewer" style="zoom:40%;" />

 ## SDF

+ Key features are curvature, skeleton, and density field.
+ Heat method on PCD is also provided in `test_sdf.py`, if use heat directions as alignment.

<table>
  <tr>
    <td align="center" width="33%">
      <img src="./md_assets/min_curv.png" alt="Image 1" width="100%"/>
      <br>
      <b>Min curv directions</b>
    </td>
    <td align="center" width="33%">
      <img src="./md_assets/skeleton.png" alt="Image 2" width="100%"/>
      <br>
      <b>Skeleton from SDF</b>
    </td>
    <td align="center" width="33%">
      <img src="./md_assets/density.png" alt="Image 3" width="100%"/>
      <br>
      <b>Density field</b>
    </td>
  </tr>
</table>


## Guidance field

+ Steamlines on surface and interior isolayers.

<table>
  <tr>
    <td align="center" width="20%">
      <img src="./md_assets/bunny_stream.png" alt="Image 1" width="100%"/>
      <br>
      <b>Bunny streamlines (1st stage)</b>
    </td>
    <td align="center" width="24%">
      <img src="./md_assets/bunny_isolayers.png" alt="Image 2" width="100%"/>
      <br>
      <b>Bunny isolayers (2nd stage)</b>
    </td>
    <td align="center" width="25%">
      <img src="./md_assets/fer_stream.png" alt="Image 3" width="100%"/>
      <br>
      <b>Fertility streamlines (1st stage)</b>
    </td>
    <td align="center" width="24%">
      <img src="./md_assets/fer_isolayers.png" alt="Image 4" width="100%"/>
      <br>
      <b>Fertility isolayers (2nd stage)</b>
    </td>
  </tr>
</table>


## Infill field

+ Density-aware or uniform, orientation-tunable infill fields. 

<table>
  <tr>
    <td align="center" width="21%">
      <img src="./md_assets/bunny_infill1.png" alt="Image 1" width="100%"/>
      <br>
      <b>Bunny density infill, beta=0</b>
    </td>
    <td align="center" width="20%">
      <img src="./md_assets/bunny_infill_uni1.png" alt="Image 2" width="100%"/>
      <br>
      <b>Bunny uniform infill, beta=0</b>
    </td>
    <td align="center" width="25%">
      <img src="./md_assets/bunny_infill2.png" alt="Image 3" width="100%"/>
      <br>
      <b>Bunny density infill, beta=90</b>
    </td>
    <td align="center" width="24%">
      <img src="./md_assets/bunny_infill_uni2.png" alt="Image 4" width="100%"/>
      <br>
      <b>Bunny uniform infill, beta=90</b>
    </td>
  </tr>
</table>


## Toolpath generation

> :unlock::key: Custom CUDA package has to be installed to enable toolpath generation. See [installation](#installation):point_up:.
>
> Note that, some vis functions here take long time to draw.

<table>
  <tr>
    <td align="center" width="28%">
      <img src="./md_assets/bunny_isocontour.png" alt="Image 1" width="100%"/>
      <br>
      <b>Bunny iso-contours</b>
    </td>
    <td align="center" width="28%">
      <img src="./md_assets/bunny_seq.png" alt="Image 2" width="100%"/>
      <br>
      <b>Bunny printing sequence</b>
    </td>
    <td align="center" width="37%">
      <img src="./md_assets/bunny_inflll.jpg" alt="Image 3" width="100%"/>
      <br>
      <b>Bunny infills</b>
    </td>
  </tr>
</table>


## Level field

+ Level field in space is to partition space, guidance field plane at adjacent levels as classification hyperplanes.

<table>
  <tr>
    <td align="center" width="47%">
      <img src="./md_assets/bunny_part.png" alt="Image 1" width="70%"/>
      <br>
      <b>Bunny level partition</b>
    </td>
    <td align="center" width="46%">
      <img src="./md_assets/fer_part.png" alt="Image 2" width="70%"/>
      <br>
      <b>Fertility level partition</b>
    </td>
  </tr>
</table>




## TV-SDF

+ TVSDF at given waypts shown in viewer (toggle off pcd). <span style="color:red">Red point</span> indicates current working waypoint.

<table>
  <tr>
    <td align="center" width="47%">
      <img src="./md_assets/bunny_tvsdf.png" alt="Image 1" width="70%"/>
      <br>
      <b>Bunny TVSDF</b>
    </td>
    <td align="center" width="48%">
      <img src="./md_assets/fer_tvsdf.png" alt="Image 2" width="70%"/>
      <br>
      <b>Fertility TVSDF</b>
    </td>
  </tr>
</table>


## Quaternion field and collision response

> This project gives a simple nozzle model. Users can prepare your nozzle model (in physical scale).

+ Differentiable collision response with TVSDF. TVSDF gradients as push directions.
+ Comparision before and after optimization to be collision-free.

<table>
  <tr>
    <td align="center" width="48%">
      <img src="./md_assets/bunny_reposes.png" alt="Image 1" width="80%"/>
      <br>
      <b>Bunny collision response</b>
    </td>
    <td align="center" width="38%">
      <img src="./md_assets/fer_response.jpg" alt="Image 2" width="80%"/>
      <br>
      <b>Fertility collision response</b>
    </td>
  </tr>
</table>



<table>
  <tr>
    <td align="center" width="27%">
      <img src="./md_assets/bunny_case1.jpg" alt="Image 1" width="100%"/>
      <br>
      <b>Bunny case 1</b>
    </td>
    <td align="center" width="29%">
      <img src="./md_assets/fer_case1.jpg" alt="Image 2" width="100%"/>
      <br>
      <b>Fertility case 1</b>
    </td>
    <td align="center" width="21%">
      <img src="./md_assets/fer_case2.jpg" alt="Image 3" width="100%"/>
      <br>
      <b>Fertility case 2</b>
    </td>
    <td align="center" width="18%">
      <img src="./md_assets/fer_case3.jpg" alt="Image 4" width="100%"/>
      <br>
      <b>Fertility case 3</b>
    </td>
  </tr>
</table>


# Others

## BibTex

```md
@article{Qu2025INF3DP,
      title={INF-3DP: Implicit Neural Fields for Collision-Free Multi-Axis 3D Printing},
      author={Qu, Jiasheng and Huang, Zhuo and Guo, Dezhao and Sun, Hailin and Lyu, Aoran and  Dai, Chengkai and Yam, Yeung and Fang, Guoxin},
      journal={ACM Transactions on Graphics (TOG)},
      note={To appear in SIGGRAPH Asia 2025},
      pages={1--18},
      year={2025},
      publisher={ACM}
      }
```

## Acknowledgements

This code is built based on the fantastic work of [SIREN](https://github.com/vsitzmann/siren). 









