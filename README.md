# HuggingFace Model Training Helper by LawTalk

## Contact
If there is any issue (any issue at all),

please contact `Wonsuk Yang (LawTalk & KAIST)`
> derrick0511@kaist.ac.kr (primary)

> emmanuel20232035@gmail.com (primary)

> wy17892022@gmail.com 

> derrick0511@nlp.kaist.ac.kr

> ws.yang@lawcompany.co.kr

I was very cautious about the Code Licence.
Yet, I am rather certain that there must be a mistake. 

Therefore, if there is any concern or an issue,
please do not hesitate to contact.

## Quick Start

```bash
$ cd ~/Downloads
$ git clone https://github.com/lawcompany/lawtalk_neural_model_helper.git
$ cd lawtalk_neural_model_helper
$ pip install -r requirements.txt
$ python train_models.py 
```
The command above should train the models according to the `hugging_face_train_params.json`, then save models at `results` dir.

If any error bothers you, please contact me (please refer to the email addresses above). I tried to cover all possible exceptions, but if it bothered you, my bad ;) sorry about that.

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
it will show a prompt in which you can type a text. The text will be given to the best model then the model will produce an output

## Parameter Settings
you can adjust the parameters in `hugging_face_train_params.json` for more various parameters to try with.
The `train_models.py` try to train with all permutations of the options, and if OOM error occurs, it logs the error then proceeds.
The parameter file contains options as follows: 

```json
{
  "hugging_face_model_name": ["klue/roberta-base", "klue/bert-base", "klue/roberta-large", "klue/roberta-small"],
  "num_train_epochs": [3],
  "per_device_batch_size": [8, 32],
  "save_steps": [50],
  "seed": [42, 4242],
  "evaluation_strategy": ["steps"],
  "gradient_accumulation_steps": [16],
  "eval_accumulation_steps": [32],
  "eval_steps": [50],
  "logging_steps": [25],
  "learning_rate": [1e-4],
  "weight_decay": [5e-2, 5e-1],
  "save_total_limit": [3],
  "load_best_model_at_end": ["True"],
  "metric_for_best_model": ["f1"],
  "label_names": [["labels"]]
}
```


## Copyright
All rights reserved, LawTalk (Law&Company)
