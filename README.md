# simdat

simdat states for "simple data analysis toolset" and is a simple tool for data analysis and machine learning.

## Installation

* Download the source code
* Add the parent directory to PYTHONPATH
```shell
$cp simdat/setdevenv . && source setdevenv
```
* To use the plotting methods with ssh or docker, copy core/matplotlibrc to ~/.config/matplotlib/

## Setup
* Include simdat in your python scripts
```python
from simdat.core import tools, ml, plot
```

## Architecture

    .
    |-- core       # Core folder for tooling files
        |-- ml.py
        |-- plot.py
        |-- tools.py
    |-- docker     # Folder of useful docker files
    |-- examples   # Files of examples
        |-- ml.json
        |-- ml_example.py
    `-- setdevenv  # Can be used for setting PYTHONPATH

Module structure see [here](https://www.dropbox.com/s/q4mn2p507gksign/simdat.jpg?dl=0)

## License
[GNU General Public License](http://www.gnu.org/licenses/)

## Author
* Name: Tammy Yang
* Email: tammy@dt42.io

