<div align="center">
    <h1> 🐍 Python Scripts </h1>
    <h2 style="font-size:1rem"> Scripts for you! </h2>
    <p>
    This repository brings together a variety of useful scripts for everyday tasks, automation, and analysis, published daily using different libraries or tools.
    <br> Each script is designed to solve common problems or streamline repetitive processes, covering everything from data analysis to productivity utilities.
    <br> <b>Explore, learn, and contribute your own scripts to help this collection grow.</b>
    </p>

<!-- Badges -->

[![Python][Python Badge]](https://www.python.org/) [![Scripts][Scripts Everyday]]() [![Status][State]]()
[![Apache][Apache]]() [![MadeWithLove][MadeWithLove]]()
[![GitHub stars][Stars]](https://github.com/YamilAyma/Daily-Python-Library)

</div>



### 🐍 Running scripts

This project contains scripts in the `scripts/` folder that you can run in two ways:

### 🚀 1. Direct execution (standalone script)

Ideal for quick tests or when you want to run the script without external usage.

**Example**

```bash
python script.py
```

🔹 Advantages:
- Instant execution.
- Requires minimal configuration, just create a folder to store the scripts.
- Necessary for scripts that use CLI, although you can skip arguments and use it as a module. [See](#📦-2-module-execution)

> [!NOTE]
> Thanks to a small internal adjustment, this script can import other modules from the project without issues even if run directly.

---

### 📦 2. Module execution

If you want to ensure Python handles scripts correctly for use in projects, use module mode:

**Example of execution**
```bash
python -m script_folder.script
```

**Example of use as a library**
```python
from scripts import wrapperscomplexipy as wc

# Use
print(wc.analyze_directory(".", threshold=15))
```

🔹 Advantages:
- Avoids issues with relative imports regarding the folder location.
- Useful in projects where auxiliary functions are required.

### 📂 Recommended project structure

Your scripts can be inside a `scripts/` folder, and this folder should contain an `__init__.py` file so Python treats it as a package:

```text
python-project/
│
├── scripts/
│   ├── __init__.py
│   ├── scripttypes.py
│   └── another_script.py
│   
└── README.md
└── ... 
```

> [!NOTE]
> `scripttypes.py` is a special file to provide documentation resources for the scripts. IT MUST BE ADDED TO SUPPORT ALL SCRIPTS PRESENTED IN THIS REPOSITORY.

<!-- SCRIPT_TABLE_START -->
# 🧩 Scripts

<details>
<summary><a href="scripts/generate_workflows_pointblank.py">1. 📘 generate_workflows_pointblank.py</a> <kbd>Show details</kbd></summary>

| Description | CLI |
|-------------|---------|
| Utilities to accelerate data validation workflows, primarily by generating baseline validation YAML files from existing data and create actions | ✅ |


| Function | Description | Category | Tags | Status |
|----------|-------------|----------|------|--------|
| `generate_baseline_yaml()` | Generates a baseline validation YAML file from a DataFrame. | generation | yaml, scaffolding, profiling | development |
| `run_validation_from_yaml()` | Executes a Pointblank validation defined in a YAML file. | execution | yaml, workflow, integration | development |


📝 **Note**: SETUP: pip install pointblank pandas pyyaml

🔗 **Links**
| Name | URL |
|------|-----|
| Pointblank GitHub Repository | [https://github.com/posit-dev/pointblank](https://github.com/posit-dev/pointblank) |
| Pointblank Documentation | [https://posit-dev.github.io/pointblank/user-guide/](https://posit-dev.github.io/pointblank/user-guide/) |
| with YAML | [https://posit-dev.github.io/pointblank/user-guide/yaml-validation-workflows.html](https://posit-dev.github.io/pointblank/user-guide/yaml-validation-workflows.html) |
| about actions | [https://posit-dev.github.io/pointblank/user-guide/actions.html](https://posit-dev.github.io/pointblank/user-guide/actions.html) |

    
---

</details>

<details>
<summary><a href="scripts/formulaic_utils.py">2. 📘 formulaic_utils.py</a> <kbd>Show details</kbd></summary>

| Description | CLI |
|-------------|---------|
| Utilities to simplify data preprocessing using Wilkinson formulas with Formulaic, creating design matrices for modeling. | ✅ |


| Function | Description | Category | Tags | Status |
|----------|-------------|----------|------|--------|
| `build_and_train_model()` | Creates a design matrix from a formula and trains a simple Linear Regression model. | modeling | scikit-learn, linear-regression, end-to-end | development |
| `create_model_matrix()` | Creates a model matrix (and outcome vector, if specified) from a DataFrame. | matrix_creation | formula, dataframe, design-matrix | development |
| `transform_train_test_split()` | Applies a formula to a train/test split, ensuring consistent encoding. | data_transformation | train-test, encoding, consistency, pipeline | development |


📝 **Note**: SETUP: pip install formulaic pandas scikit-learn

🔗 **Links**
| Name | URL |
|------|-----|
| Formulaic GitHub Repository | [https://github.com/matthewwardrop/formulaic](https://github.com/matthewwardrop/formulaic) |

    
---

</details>

<details>
<summary><a href="scripts/image_quote_generator.py">3. 📘 image_quote_generator.py</a> <kbd>Show details</kbd></summary>

| Description | CLI |
|-------------|---------|
| Generates a batch of images by overlaying text quotes onto background images, with text wrapping and customization options. | ✅ |


| Function | Description | Category | Tags | Status |
|----------|-------------|----------|------|--------|
| `generate_quote_images()` | Generates a series of images with quotes overlaid. | generation | image, text, quotes | development |


📝 **Note**: SETUP: pip install Pillow. You also need a .ttf or .otf font file.

🔗 **Links**
| Name | URL |
|------|-----|
| Pillow Documentation | [https://pillow.readthedocs.io/en/stable/](https://pillow.readthedocs.io/en/stable/) |
| Google Fonts (for .ttf files) | [https://fonts.google.com/](https://fonts.google.com/) |

    
---

</details>

<details>
<summary><a href="scripts/wrapperscomplexipy.py">4. 📘 wrapperscomplexipy.py</a> <kbd>Show details</kbd></summary>

| Description | CLI |
|-------------|---------|
| How understandable is your codebase? Use complexipy functions to analyze the cognitive complexity of your code. | ❌ |


| Function | Description | Category | Tags | Status |
|----------|-------------|----------|------|--------|
| `analyze_directory()` | Analyze the cognitive complexity of the files in a directory. | file_analysis | complexity, file | development |
| `compare_complexities()` | Compares the cognitive complexity of two code snippets. | code_analysis | complexity, code | development |
| `compare_complexity_git()` | Compares the complexity of a Python file between the current state and a previous Git commit. | file_analysis | complexity, git | development |
| `to_csv()` | Generates a CSV report from a Python file's complexity analysis. | file_analysis | complexity, csv | development |
| `visualize_complexity_with_matplotlib()` | Visualizes the cognitive complexity of a Python file using a bar chart. | file_analysis | complexity, visualization | development |


📝 **Note**: SETUP: pip install complexipy matplotlib

🔗 **Links**
| Name | URL |
|------|-----|
| complexipy Documentation | [https://rohaquinlop.github.io/complexipy/#](https://rohaquinlop.github.io/complexipy/#) |

    
---

</details>

<details>
<summary><a href="scripts/darts_utils.py">5. 📘 darts_utils.py</a> <kbd>Show details</kbd></summary>

| Description | CLI |
|-------------|---------|
| A collection of utility functions to simplify forecasting, backtesting, and anomaly detection with Darts.    
                    Darts is a Python library designed for user-friendly time series forecasting and anomaly detection. It provides a unified API for various models, from classical statistical methods to deep learning architectures.
                     | ✅ |


| Function | Description | Category | Tags | Status |
|----------|-------------|----------|------|--------|
| `detect_anomalies_by_quantile()` | Detects anomalies using a QuantileDetector based on the series' own history. | anomaly_detection | outliers, detection, quantile, statistics | development |
| `evaluate_model_backtesting()` | Performs historical backtesting to evaluate a model's performance. | evaluation | backtesting, metrics, MAPE, RMSE | development |
| `quick_forecast_and_plot()` | Trains a simple forecasting model, makes a prediction, and saves a visualization. | forecasting | forecast, visualization, ARIMA, ExponentialSmoothing | development |


📝 **Note**: SETUP: pip install darts pandas matplotlib

🔗 **Links**
| Name | URL |
|------|-----|
| Darts GitHub Repository | [https://github.com/unit8co/darts](https://github.com/unit8co/darts) |
| Demand Forecasting with Darts: A Tutorial | [https://towardsdatascience.com/demand-forecasting-with-darts-a-tutorial-480ba5c24377/](https://towardsdatascience.com/demand-forecasting-with-darts-a-tutorial-480ba5c24377/) |
| Darts Documentation | [https://unit8co.github.io/darts/](https://unit8co.github.io/darts/) |
| Resources | [https://unit8.com/resources/darts-time-series-made-easy-in-python/](https://unit8.com/resources/darts-time-series-made-easy-in-python/) |

    
---

</details>


<!-- SCRIPT_TABLE_END -->

**🚀 Simply download any script that interests you and integrate it into your project!**
Some scripts offer CLI mode and include documentation within their files.


# TODO
- [X] Tool for automate script documentation
- [ ] Website for scripts
- [ ] Ideas from community 

---

Give a ⭐ if you find it useful.

<!-- LINKS -->
[Python Badge]: https://img.shields.io/badge/python-3.10%2B-blue?logo=python
[Scripts Everyday]: https://img.shields.io/badge/Scripts-Daily-ff69b4?logo=github
[State]: https://img.shields.io/badge/Status-Active-brightgreen?style=flat
[Apache]: https://img.shields.io/badge/License-MIT-yellow
[MadeWithLove]: https://img.shields.io/badge/Made%20with-%E2%9D%A4-red
[Stars]: https://img.shields.io/github/stars/YOUR-USER/YOUR-REPO?style=social