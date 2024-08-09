# 🧠 HerBrain
HerBrain uses AI for quantifying changes in the female brain during menstruation, pregnancy, and menopause.

## 🎬 HerBrain App Demo ##

[![Demo](/images/HerBrainDemo_thumbnail.png)](https://youtu.be/zUucJbwaaO4)

## 🎤 Our Public Talk on Womens' Brain Health and AI ##

[![BBB Talk](/images/bbb_thumbnail.png)](https://youtu.be/BsdNQUcwb1M)
Visual on thumbnail slide taken from: Caitlin M Taylor, Laura Pritschet, and Emily G Jacobs. The scientific body of knowledge–whose body does it serve? A spotlight on oral contraceptives and women’s health factors in neuroimaging. Frontiers in neuroendocrinology, 60:100874, 2021.

## 🤖 Installing HerBrain

1. Clone a copy of `herbrain` from source:
```bash
git clone https://github.com/geometric-intelligence/herbrain
cd herbrain
```
2. Create an environment with python >= 3.10
```bash
conda create -n herbrain python=3.10
```
3. Install `herbrain` in editable mode (requires pip ≥ 21.3 for [PEP 660](https://peps.python.org/pep-0610/) support):
```bash
pip install -e .[all]
```
4. Install pre-commit hooks:
```bash
pre-commit install
```

## 🌎 Bibtex ##
If this code is useful to your research, please cite:

```
@misc{myers2023geodesic,
      title={Geodesic Regression Characterizes 3D Shape Changes in the Female Brain During Menstruation},
      author={Adele Myers and Caitlin Taylor and Emily Jacobs and Nina Miolane},
      year={2023},
      eprint={2309.16662},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## 🏃‍♀️ How to Run the Code ##

1. Choose whether to launch pregnancy or menstrual cycle app. Pregnancy app is in the `project_pregnancy` folder, and menstrual app is in the `project_menstrual` folder.
2. Run the `main_3_dash_app.py` file by running:

```
python main_3_dash_app.py
```

## 👩‍🔧 Authors ##
[Adele Myers](https://ahma2017.wixsite.com/adelemyers)

[Nina Miolane](https://www.ninamiolane.com/)

## How to Set up Your Environment

```shell
$ conda create -n herbrain --file conda-linux-64.lock
$ conda activate herbrain
$ poetry install --no-root
```
We use `--no-root` because we don't have a module named `herbrain`
If you are on Mac, make and use `conda-osx-64.lock` instead.

# Dev

Only run if changes are made to the environment files.

To recreate the conda lock, after modifying conda.yaml:
```shell
pip install conda-lock
make conda-linux-64.lock
```
Note that you may need to install conda-lock not in your base env.

To recreate the poetry lock, after modifying pyproject.toml:
```shell
make poetry.lock
```

