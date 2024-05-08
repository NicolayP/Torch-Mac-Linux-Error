# Torch-Mac-Linux-Error

This repo contains everything to reproduce a prediction error between a mac an a linux machine.

## Context

While trying to learn an autoregressive model on linux we noticed that the network wasn't learning anything. We randomly tried runing it on a mac we noticed outstanding performances.
The results are consistant when executed on the same os. We give the results when the same weigts are used on both machines here after:

### Linux

The expected result on a linux machine are as follows

![linux_vx](assets/linux_vx_comparison.pdf)
![linux_vy](assets/linux_vy_comparison.pdf)
![linux_vz](assets/linux_vz_comparison.pdf)

The expected 1st prediction when runing the `run.py` script:

```bash
tensor([-192.7449, 255.3847, 153.4133, 605.0937, 1240.0849, 2499.9711], dtype=torch.float64)
```

### Mac

![mac_vx](assets/mac_vx_comparison.pdf)
![mac_vy](assets/mac_vy_comparison.pdf)
![mac_vz](assets/mac_vz_comparison.pdf)

The expected 1st prediction when runing the `run.py` script:

```bash
tensor([-0.0719, -0.0403, -0.0102, -0.0413,  0.1258, -0.0873], dtype=torch.float64)
```

## Install 

There is a python environment file that can be installed with conda. 

`conda create --name env_name -f environment.yml`

If the install fails for some reason, the error has reproduced using torch 2.0-2.4 so feel free to try it with any of the versions.


## Running

There are two scripts to generate the results:

1. `run.py` Loads the model trained on the mac machine as well as the training data and runs the model.
2. `train.py` Runs the training of the model with a fixed seed to get the same results as on the mac machine.

Both script will also generate the results in a pdf format for quick visual representaiton.

## More info

What has been tried.

- Tried on 3 different Linux machines 18.04, 20.04, 22.04.
- Tried with different pytorch version 2.0, 2.1, 2.2, 2.3, 2.4.
- Looked at different Linear algebra backend (MKL, OPENBLAS, ADVANCED) on mac and linux.
- Reproduced the results on 2 different MACs machine