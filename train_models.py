from torch.utils.data import Dataset
from torch import tensor
from utils.base_handler import HuggingFaceHelper


class HuggingFaceModelTrainingScheduler(HuggingFaceHelper):
    def __init__(self, gpu_id=0):
        """
        Args:
            gpu_id: an int
        """
        super().__init__(gpu_id=gpu_id)
        from utils.misc import CudaOperators
        env_params = CudaOperators.load_env_params()
        self.dataset_name = env_params['dataset_name']
        self.task_name = env_params['task_name']
        self.early_stop_patience = 2
        self.id2label = self.get_mapping_from_label_id_to_label_str()
        self.param_sample = {
            "hugging_face_model_name": "klue/roberta-small", "num_train_epochs": 20, "per_device_train_batch_size": 32,
            "per_device_eval_batch_size": 32, "save_steps": 50, "seed": 42, "evaluation_strategy": 'steps',
            "gradient_accumulation_steps": 16, "eval_accumulation_steps": 32,
            "eval_steps": 50, "logging_steps": 25, "learning_rate": 1e-4, "weight_decay": 5e-2, "save_total_limit": 3,
            "load_best_model_at_end": "True", "metric_for_best_model": 'f1', "label_names": ['labels']
        }

    def __call__(self, param):
        """
        Args:
            param: a dict

        Returns:
            None
        """
        if not param:
            param = self.param_sample
        param = self.param_fix_for_cpu_case(param=param)
        self.auto_train(param=param)

    @staticmethod
    def param_fix_for_cpu_case(param):
        """
        Args:
            param: a dict

        Returns:
            a dict
        """
        from torch.cuda import is_available
        if is_available():
            param['fp16'] = True
            param['no_cuda'] = False
        else:
            param['fp16'] = False
            param['no_cuda'] = True
        return param

    @staticmethod
    def get_mapping_from_label_id_to_label_str():
        from utils.data_loader import JSONLoader
        from utils.constant import Directories
        id2label = JSONLoader(dir_=Directories.LAW_SERVICE_MAPPING_DIR)()
        return id2label

    def get_customized_metric(self, pred):
        labels, predictions = pred.label_ids, pred.predictions.argmax(-1)
        return self.get_accuracy_and_f1_for_predictions_and_labels(predictions=predictions, labels=labels)

    @staticmethod
    def get_accuracy_and_f1_for_predictions_and_labels(predictions, labels):
        """
        Args:
            predictions: a list
            labels: a list

        Returns:
            a dict that has a format of {"accuracy": a float, "f1": a float}
        """
        from datasets import load_metric
        to_return = {
            "accuracy": load_metric(path='accuracy').compute(
                predictions=predictions, references=labels
            )['accuracy'],
            "f1": load_metric(path='f1').compute(
                predictions=predictions, references=labels, average="macro"
            )['f1']
        }
        return to_return

    def get_model_file_dir_based_on_time(self):
        """ Returns: a str """
        from datetime import datetime
        time_now = str(datetime.now())
        time_now = time_now.replace("/", "_").replace(" ", "_")
        return f"{self.train_model_dir_root}/model_{time_now}"

    @staticmethod
    def split_original_dataset_into_train_test_datasets(original_dataset):
        """
        Args:
            original_dataset: a Dataset

        Returns:
            a tuple of (Dataset, Dataset)
        """
        dict_ = original_dataset.train_test_split(test_size=0.1, shuffle=False)
        return dict_['train'], dict_['test']

    def log_training_info(self, model_dir, param):
        """
        Args:
            model_dir: a str
            param: a dict

        Returns:
            None
        """
        import os
        from copy import deepcopy
        from utils.data_loader import JSONSaver
        if os.path.isdir(self.log_dir_root):
            pass
        else:
            os.mkdir(self.log_dir_root)
        model_name = model_dir.split("/")[-1]
        log_dir = f"{self.log_dir_root}/{model_name}.opt"
        param_to_save = deepcopy(param)
        param_to_save['output_dir'] = model_dir
        JSONSaver(dir_=log_dir, data_=param_to_save)()

    def train_model(self, param, train_data, test_data, model):
        """
        Args:
            param: a dict
            train_data: a Dataset
            test_data: a Dataset
            model: a Model

        Returns:
            None
        """
        from transformers import TrainingArguments, EarlyStoppingCallback, Trainer
        training_args = TrainingArguments(**param)
        trainer = Trainer(
            model=model, args=training_args, train_dataset=train_data,
            eval_dataset=test_data, compute_metrics=self.get_customized_metric,
            callbacks=[EarlyStoppingCallback(
                early_stopping_patience=self.early_stop_patience
            )]
        )
        trainer.train(resume_from_checkpoint=False)

    @staticmethod
    def remove_model_name_from_param_and_set_model_name_and_tokenizer(param):
        """
        Args:
            param: a dict that contains `model_name` as a key

        Returns:
            a tuple of (str, dict) where the str is the model name
            and the dict is the param dict that does not contain `model_name` as a key
        """
        model_name = param["hugging_face_model_name"]
        del param["hugging_face_model_name"]
        return model_name, param

    @staticmethod
    def add_tensor_dimension_helper_to_the_dataset(dataset, tokenizer):
        """
        Args:
            dataset: a Dataset
            tokenizer: a Tokenizer

        Returns:
            a Dataset
        """
        labels = [x['laws_service_id'] for x in dataset]
        dataset = TensorDimensionHelperAddedDataset(
            data=dataset, labels=labels, tokenizer=tokenizer
        )
        return dataset

    def get_train_and_test_data(self, use_original_data_format, tokenizer):
        """
        args:
            use_original_data: a bool
            tokenizer: a Tokenizer

        Returns:
             a tuple of (Dataset, Dataset)
        """
        from datasets import load_dataset
        original_dataset = load_dataset(
            self.dataset_name, self.task_name, split='train'
        )
        train_data, test_data = self.split_original_dataset_into_train_test_datasets(original_dataset=original_dataset)
        if not use_original_data_format:
            train_data, test_data = [
                self.add_tensor_dimension_helper_to_the_dataset(dataset=x, tokenizer=tokenizer)
                for x in [train_data, test_data]
            ]
        return train_data, test_data

    def auto_train(self, param):
        """
        Args:
            a dict

        Returns:
            None
        """
        train_data, test_data, model, model_file_dir, training_parameters, hugging_face_model_name =\
            self.get_data_along_with_the_model_info_and_training_parameters(param=param)
        self.train_model_and_save_checkpoints_if_oom_does_not_happen(
            train_data=train_data, test_data=test_data, model=model, model_file_dir=model_file_dir,
            training_parameters=training_parameters, hugging_face_model_name=hugging_face_model_name
        )

    def train_model_and_save_checkpoints_if_oom_does_not_happen(self, train_data, test_data, model, model_file_dir,
                                                                training_parameters, hugging_face_model_name):
        """
        Args:
            train_data: a HF Dataset
            test_data: a HF Dataset
            model: a HF Model
            model_file_dir: a str
            training_parameters: a dict
            hugging_face_model_name: a str

        Returns:
            None
        """
        def update_oom_info(dict_to_update, hf_model_name, oom_happened, oom_message):
            """
            Args:
                dict_to_update: a dict
                hf_model_name: a str
                oom_happened: a bool
                oom_message: a str

            Returns:
                a dict
            """
            dict_to_update['hugging_face_model_name'] = hf_model_name
            dict_to_update['whether_OOM_happened'] = oom_happened
            dict_to_update['OOM_message'] = oom_message
            return dict_to_update

        try:
            self.train_model(
                param=training_parameters, train_data=train_data, test_data=test_data, model=model
            )
            training_parameters = update_oom_info(training_parameters, hugging_face_model_name, False, "")
            self.log_training_info(model_dir=model_file_dir, param=training_parameters)
        except RuntimeError as e:
            if "CUDA out of memory. " in str(e):
                print(f"[WARNING] OOM happened, saving the OOM error message at the training log (.opt) file")
                training_parameters = update_oom_info(training_parameters, hugging_face_model_name, True, str(e))
                self.log_training_info(model_dir=model_file_dir, param=training_parameters)
            else:
                raise RuntimeError

    def get_data_along_with_the_model_info_and_training_parameters(self, param):
        """
        Args:
            param: a dict

        Returns:
            a tuple of (train_data, test_data, model, model_file_dir, training_parameters, hugging_face_model_name)
            which is (a Dataset, a Dataset, a Model, a str, a dict, a str)
        """
        from copy import deepcopy
        from torch import tensor, long
        from transformers import AutoTokenizer
        from transformers import AutoModelForSequenceClassification
        hugging_face_model_name, param = self.remove_model_name_from_param_and_set_model_name_and_tokenizer(param=param)
        tokenizer = AutoTokenizer.from_pretrained(hugging_face_model_name)
        train_data, test_data = self.get_train_and_test_data(use_original_data_format=False, tokenizer=tokenizer)
        training_parameters = deepcopy(param)
        model_file_dir = self.get_model_file_dir_based_on_time()
        possible_labels = tensor(list(set(train_data.labels))).type(long)
        model = AutoModelForSequenceClassification.from_pretrained(
            hugging_face_model_name, num_labels=len(possible_labels)
        )
        training_parameters['output_dir'] = model_file_dir
        return train_data, test_data, model, model_file_dir, training_parameters, hugging_face_model_name


class TensorDimensionHelperAddedDataset(Dataset):
    def __init__(self, data, labels, tokenizer):
        self.data = data
        self.labels = labels
        self.tokenizer = tokenizer

    def __getitem__(self, i):
        data = self.tokenizer(
            self.data[i]['fact'], padding='max_length', truncation='longest_first', max_length=512
        )
        data['labels'] = tensor(self.labels[i])
        return data

    def __len__(self):
        return len(self.labels)


class Trainer:
    def __init__(self):
        from utils.misc import CudaOperators
        self.training_params_dir = CudaOperators.load_env_params()['target_training_params_dir']

    def run_auto_train(self):
        """ Returns: None """
        scheduler = HuggingFaceModelTrainingScheduler()
        params = self._get_auto_train_params()
        for param in params:
            scheduler(param=param)

    def _get_auto_train_params(self):
        """
        Returns:
            a list of dict where each dict has the key of str and the value of an int or a str or a list
        """
        from utils.data_loader import JSONLoader
        param_dir = self.training_params_dir
        param_options = JSONLoader(dir_=param_dir)()
        param_option_list = self._get_param_option_list_from_param_options(
            param_options=param_options
        )
        param_option_list = [self._clean_up_param_options(x) for x in param_option_list]
        return param_option_list

    @staticmethod
    def _get_param_option_list_from_param_options(param_options):
        """
        Args:
            param_options: a dict that has a key of str and the value of the list
        """
        from itertools import product
        key_list, list_of_val_list = [], []
        for key_, val_list in param_options.items():
            key_list.append(key_)
            list_of_val_list.append(val_list)
        param_list_to_return = []
        all_possible_params_accumulated = list(product(*list_of_val_list))
        for list_of_possible_params in all_possible_params_accumulated:
            assert len(key_list) == len(list_of_possible_params)
            new_param = dict()
            for idx, param_value in enumerate(list_of_possible_params):
                new_param[key_list[idx]] = param_value
            param_list_to_return.append(new_param)
        return param_list_to_return

    @staticmethod
    def _clean_up_param_options(param_option):
        """
        Args:
            param_option: a dict where the key is str and the value is an int or a str or a list

        Returns:
            a dict where the key is str and the value is an int or a str or a list
        """
        param_option["per_device_train_batch_size"] = param_option["per_device_batch_size"]
        param_option["per_device_eval_batch_size"] = param_option["per_device_batch_size"]
        del param_option["per_device_batch_size"]
        return param_option


if __name__ == "__main__":
    Trainer().run_auto_train()
