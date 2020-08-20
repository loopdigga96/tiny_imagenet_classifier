### Project structure
```
- database "contains all necessary functions for storing requests in sqlite"
- modules "contains model and dataset"
- service.py "contains FastAPI wrapper on model"
- train_solution.py "contains whole pipeline"
```
### How to install
- Create virtualenv with **python 3.7**
- `pip install -r requirements.txt`

### How to train
- `python train_solution.py`
- pytorch-lightning is used for training
- In results folder will be created folder with experiments with *checkpoints* and *tb_logs*
- to see metrics run `tensorboard --log_dir results/name_of_experiment/tb_logs`
- after training; csv file with test predictions will be created

### Improvement steps
- use pre-trained model (+28% acc)
- resize image while using pre-trained model
- use better backbone (efficient net) 
- use model with bigger capacity (**b0** ef_net vs **b4**)
- use normalization params from original ImageNet dataset
- change optimizer to more robust Adam
- lower lr while doing fine tuning of pretrained model
- add scheduler ReduceLROnPlateau(patience=1, min_lr=1e-6, mode='min', factor=0.3)
- add label smoothing stabilize training (+1% acc)


### Possible improvements (that i did not tried because it would take much time on my machine):
- make ensemble of models (usually it helps), starting from simple probabilities averaging and take predictions from different efficient nets
- make tricky augmentations for example from ablumentations
- try Focal Loss
- increase capacity of model b7 efficient net
- use SOTA model on ImageNet dataset according to [paperswithcode](https://paperswithcode.com/sota/image-classification-on-imagenet) for example efficient net b7  noisy student
- make even more experiments with optimizer and schedulers. For example use LR Cyclic scheduler, lr warm up and decay
- use Test Time Augmentations


## Result:
- model: efficeint_netb4 with label smoothing, Adam, ReduceLROnPlateau, EarlyStopping
- final val accuaracy: 0.7544
- final csv with test predictions: [download](https://drive.google.com/file/d/1klFWpRssLdTUMe6KfApWZQhw-_PdtBRR/view?usp=sharing)


## How to run service (FastAPI):
- [download](https://drive.google.com/drive/folders/1jWtrVRMjFbKbSpfZkkI6GjysPnCbh8j4?usp=sharing) model and place it to results_folder;  service will use this checkpoint in inference
- `uvicorn --host 0.0.0.0 service:app --reload`
- POST: `127.0.0.1:8000/predict/` 
    - `curl -X POST "http://127.0.0.1:8000/predict/" -H  "accept: application/json" -H  "Content-Type: multipart/form-data" -F "file=@path_to_image.jpg;type=image/webp"`
- GET: `127.0.0.1:8000/get_all/`
    - `curl -X GET "http://127.0.0.1:8000/get_all/" -H  "accept: application/json"`
- Shows nice UI where you can check all endpoints `127.0.0.1:8000/docs`
