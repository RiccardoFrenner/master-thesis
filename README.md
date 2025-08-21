Fork of the BayesFlow python library to implement correction methods for ABI for my master thesis.

Most of the code of the thesis is in `my_stuff`. Only little code of the BayesFlow library had to change. These changes can be found in the `bayesflow/networks` folder. For a detailed changelog, see the git history and look for changes by me (RiccardoFrenner)

# Reproducibility
0. create a virtual environment `python -m venv ./venv & source ./venv/bin/activate`
1. run `pip install -e .` This installs the BayesFlow library
2. run `pip install "keras>=3.9" tensorflow numpy pandas matplotlib seaborn tqdm scipy scikit-learn ipykernel`
## SIR
1. run `my_stuff/code/main.ipynb`
