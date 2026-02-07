# Setup

All the tutorial and homework content is within the current directory of the repository.
Homework submission happens via ILIAS, as a Jupyter Notebook answering the assignment questions and containing the code needed to produce the answers.

Before working on any lab task, please follow the following steps to get up and running:
- Ensure [Python 3](https://www.python.org/downloads/) is installed (preferably a version from 3.9 to 3.13, but older ones should work by lowering some dependency versions in the requirements file).
- Have a [VS Code](https://code.visualstudio.com/Download) installation with the [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python) and [Jupyter](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter) extension packs (other IDEs like [PyCharm](https://www.jetbrains.com/pycharm/download/) also work, but this guide does not contain specific configuration files/steps for them).
- From the root directory of this repository, execute `python3 -m venv lab/.venv`.
- Open the repository in VS Code, and [create a new terminal](https://code.visualstudio.com/docs/terminal/basics) (checking that current line starts with `(.venv)` to signal the correct Python environment got activated).
- Move to the `lab` directory and execute `pip install -r requirements.txt` in the terminal.
