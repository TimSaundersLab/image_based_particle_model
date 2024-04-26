# image_based_particle_model

![FRAP_panel_final-03](https://github.com/Yi-Ting-Loo/image_based_particle_model/assets/91601788/6218b12a-5238-4310-b60f-f5616266c9c5)


This repo is an image-based particle model Monte Carlo simulation for the article by Zhu and Loo $\textit{et. al.}$:

### Receptor binding and tissue architecture explains the morphogen local-to-global diffusion coefficient transition

Shiwen Zhu<sup>1</sup>, Yi Ting Loo <sup>2,3</sup>, Sapthaswaran Veerapathiran<sup>1</sup>, Tricia Y. J. Loo<sup>3,5</sup>, Bich Ngoc Tran<sup>1</sup>, Cathleen Teh<sup>1</sup>, Jun Zhong<sup>1,5</sup>, Paul T. Matsudaira<sup>1</sup>, Timothy E Saunders<sup>1,3</sup>, and Thorsten Wohland<sup>1,4,7</sup>.

<sup>1</sup>NUS Centre for Bio-Imaging Science, Department of Biological Sciences, National University of Singapore, Singapore 117558
<sup>2</sup>Mathematics Institute, University of Warwick, Coventry CV4 7AL, United Kingdom
<sup>3</sup>Warwick Medical School, University of Warwick, Coventry CV4 7AL, United Kingdom
<sup>4</sup>Department of Chemistry, National University of Singapore, Singapore 117543
<sup>5</sup>Mechanobiology Institute, National University of Singapore, Singapore 117411
<sup>6</sup>Institute of Molecular and Cell Biology, A*STAR, Singapore 138673
<sup>7</sup>Institute of Digital Molecular Analytics and Science, National University of Singapore, Singapore 636921

### Recommended set-up for project

Create a virtual environment using:
```bash
conda create -n ImageDiff python=3.9.12
conda activate ImageDiff
```
Add `conda-forge` channel required for some libraries:
``` bash
conda config --env --add channels conda-forge
```
Install package versions listed in `requirements.txt`:
```bash
conda install --file requirements.txt
```
Clone the repo:
```bash
git clone https://github.com/TimSaundersLab/image_based_particle_model.git
```

### Folders description
`binary_images` contains the realistic images of extracellular spaces in the zebrafish brain tissue architecture.
`toy_data/toy_OT` is a smaller stack of binary images to test the simulations.
`particle_models` contains the main functions for FRAP and FCS image-based particle modelling.
`scripts` contains python scripts with example usages of the functions.
