# Research Engineer test task: Image Classifier for Tiny ImageNet
## Introduction
Dear applicant! We are asking you to implement a thoughtful pipeline for a rather simple toy task.

You are given the dataset from the Tiny ImageNet Challenge, which is the default final project for Stanford CS231N course. It runs similar to the ImageNet challenge (ILSVRC). The goal is to solve the Image Classification problem as good as possible, and then deploy your model as a REST endpoint. The deployed microservice should save information on each request to a file or a database or somewhere else , local or in a separate microservice (so that one could monitor the model health using these metrics later):
- duration of model inference,
- inference result,
- model confidence (if applicable),
- maybe some other metrics you find useful.

In this task, we encourage you to apply best practices of engineering and demonstrate your skills of rapid prototyping and building reliable and maintainable pipelines. You are free to focus more on the part of the model lifecycle you find more interesting (model quality or model inference aspects).

We provide you with [a simple pytorch baseline](baseline.ipynb), feel free to use it or to design the entire solution from scratch. We are not restricting you with the frameworks, you can install any package you need and organise your files however you want. Just please make sure to provide all the sources, checkpoints, logs, visualisations etc. It is desirable, but not demanded, that you use [Neu.ro Platform](https://neu.ro) for the development as it would ease the process of your further on-boarding. If you decide to use our platform setup, it is all already taken care of. We will just review the artifacts of your work on Neu.ro Storage. Otherwise, it is your responsibility.

To add some measurable results to the task, your final goal will be to achieve the best accuracy on the provided test split of the Tiny ImageNet dataset. Also, you are expected to show all the visualizations you find necessary alongside the final evaluation metrics. We already took care of a tensorboard setup for you, so you can track some of your plots there.

We would emphasize once again that your goal is to show your best practices as a researcher and developer in approaching deep learning tasks. Good luck!

# Development
For your convenience, the team has created the environment using [Neu.ro Platform](https://neu.ro), so you can jump into problem-solving right away.

# References

* [Recipe for Training Neural Networks, A. Karpathy](https://karpathy.github.io/2019/04/25/recipe/)

## Quick Start

Sign up at [neu.ro](https://neu.ro) and setup your local machine according to [instructions](https://neu.ro/docs).
 
Then run:

```shell
neuro login
make setup
make jupyter
```

See [Help.md](HELP.md) for the detailed Neuro Project Template Reference.

## Developing Locally

### With `cpu` support
To run `jupyter-notebook` locally in `Docker` container use the following command 
in the root dir of the project:
```shell
docker run -it --rm \
           -p 8889:8889 \
           -v $(pwd)/data:/project/data \
           -v $(pwd)/code:/project/code \
           -v $(pwd)/notebooks:/project/notebooks \
           -v $(pwd)/results:/project/results \
           neuromation/base:v1.4 jupyter-notebook --no-browser --port 8889 --ip=0.0.0.0 --allow-root --notebook-dir=/project/notebooks
```

To start a container with `bash` as `entrypoint` use:
 
```shell
docker run -it --rm \
           -p 8889:8889 \
           -v $(pwd)/data:/project/data \
           -v $(pwd)/code:/project/code \
           -v $(pwd)/notebooks:/project/notebooks \
           -v $(pwd)/results:/project/results \
           --entrypoint /bin/bash \
           neuromation/base:v1.4
```

### With `gpu` support

If you have installed `nvidia-docker v2` you can use the previous commands with 
`gpu` support. You just need to add `--gpus all` parameter in `docker run` command.


## Developing with Neu.ro Platform

```
$ pip install -U neuromation
$ neuro login
$ make setup             # prepare remote environment using files `requirements.txt` and `apt.txt`
$ make upload-notebooks  # upload notebooks to the Storage 
$ make jupyter           # run a jupyter notebook with specific volume mounts
$ make inference         # (not yet completed) run inference in a Neu.ro Job
```

If you decide to run model inference inside a Neu.ro Job (see target `inference` in `Makefile`), you might want to expose your server's HTTP port (`neuro run ... --http 8080`, see variable `INFERENCE_PORT`) and disable HTTP Authentication (`neuro run ... --no-http-auth`) -- these are already set up in target `inference`. The job's command should be stored in variable `INFERENCE_CMD`.

Note also parameter `--life-span` of `neuro run`: it defines the maximum job's timeout, for example `--life-span 1d` means that the job will be killed after 1 day of being active, `--life-span 1h` will kill the job after 1 hour. Use explicit `--life-span=0` to disable this timeout.
