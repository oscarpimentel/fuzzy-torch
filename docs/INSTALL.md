# Install

## Enviroment

I recommend to use Anaconda environments with Python 3.7
You can create a new environment as:
```bash
conda create -n {myenv} python=3.7.
```

Then, to enter the enviroment:
```bash
conda activate {myenv}
```

Also, to remove an env:
```bash
conda remove -n {myenv} --all
```

## Classic Installations
### Jupyer-Notebook

A very good idea is to install jupyter-notebook as soon as the new enviroment is created.
To install:
```bash
conda install -c conda-forge notebook
```

## References
- https://www.anaconda.com/distribution/
- https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html