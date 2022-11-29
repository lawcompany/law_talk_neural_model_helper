class HuggingFaceModelTrainingSchedulerTester:
    def __init__(self):
        self.target_training_params_dir = "./hugging_face_train_params.json"

    def run_auto_train(self):
        """ Returns: None """
        from backend.src.hugging_face_model_training_scheduler import HuggingFaceModelTrainingScheduler
        scheduler = HuggingFaceModelTrainingScheduler()
        params = self._get_auto_train_params()
        for param in params:
            scheduler(param=param)

    def _get_auto_train_params(self):
        """
        Returns:
            a list of dict where each dict has the key of str and the value of an int or a str or a list
        """
        from backend.src.basic_tools.data_loader import JSONLoader
        param_dir = self.target_training_params_dir
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