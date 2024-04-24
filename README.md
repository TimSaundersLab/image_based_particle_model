# image_based_particle_model

![image](https://github.com/Yi-Ting-Loo/image_based_particle_model/assets/91601788/2e859787-c773-4d65-ad13-99726565de3a)


This repo is an image-based particle model Monte Carlo simulation for the article by Zhu and Loo $\textit{et. al.}$:

### Zebrafish brain $\textit{in silico}$ architecture recapitulates morphogen local-to-global diffusion coefficient transition

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

