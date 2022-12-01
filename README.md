# HuggingFace Model Training Helper by LawTalk

## Contact
If there is any issue (any issue at all),

please contact `Wonsuk Yang (LawTalk & KAIST)`
> ws.yang@lawcompany.co.kr (primary)

> derrick0511@kaist.ac.kr (primary)

> emmanuel20232035@gmail.com

> wy17892022@gmail.com 

> derrick0511@nlp.kaist.ac.kr


I was very cautious about the Code Licence.
Yet, I am rather certain that there must be a mistake. 

Therefore, if there is any concern or an issue,
please do not hesitate to contact.

## Quick Start
### Prerequisites
You should install PyTorch before the Quick Setup.
Please visit `https://pytorch.org/get-started/locally/` for the right version of PyTorch that suits your CUDA.

You may check whether it was successfully installed by checking the following commands.
If it returns `True` then you are good to go.

```python
from torch.cuda import is_available
print(is_available())
```

### Quick Setup
```bash
$ cd ~/Downloads
$ git clone https://github.com/lawcompany/law_talk_neural_model_helper.git
$ cd law_talk_neural_model_helper
$ pip install -r requirements.txt
```

### Quick Training
```bash
$ python train_models.py 
```

The command above should train the models according to the `parameters/model_parameters.json`, then save models at `results` dir.

If any error bothers you, please contact me (please refer to the email addresses above). I tried to cover all possible exceptions, but if it bothered you, my bad; terribly sorry about that.

### Evaluation (After the Training is Done)
```bash
$ python eval_models.py
```
Automatically eval the models based on the automatically split dataset (valid data) then it will generate `results/logs/evaluation_summary_~.json`.
The summary file contains the evaluation results of the models, where the best model comes first.

### User Interaction (After the Training is Done)
```bash
$ python user_interaction.py
```
It will show a prompt in which you can type a text. The text will be given to the best model then the model will produce an output

## Parameter Settings
You can adjust the parameters in `hugging_face_train_params.json` for more various parameters to try with.
The `train_models.py` try to train with all permutations of the options, and if OOM error occurs, it logs the error then proceeds.
The parameter file contains options as follows: 

```json
{
  "hugging_face_model_name": ["klue/bert-base", "klue/roberta-large", "klue/roberta-small", "klue/roberta-base"],
  "num_train_epochs": [3, 5, 10, 15],
  "per_device_batch_size": [8, 16, 32],
  "save_steps": [50],
  "seed": [42, 4242, 424242],
  "evaluation_strategy": ["steps"],
  "gradient_accumulation_steps": [2, 4, 8, 16],
  "eval_accumulation_steps": [32],
  "eval_steps": [50],
  "logging_steps": [25],
  "learning_rate": [1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
  "weight_decay": [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2],
  "save_total_limit": [3],
  "load_best_model_at_end": ["True"],
  "metric_for_best_model": ["f1"],
  "label_names": [["labels"]]
}
```


## Copyright
All rights reserved, LawTalk (Law&Company)
