from backend.src.hugging_face_base_scheduler import HuggingFaceBaseClass


class HuggingFaceModelEvaluationSummarizer(HuggingFaceBaseClass):
    def __init__(self):
        super().__init__()

    def __call__(self):
        return self.main()

    def main(self):
        summary_results = self.get_summary_results()
        self.save_summary_results(summary_results=summary_results)

    def save_summary_results(self, summary_results):
        """
        Args:
            summary_results: a list of dict where each dict has the format of
                             {
                                 "parameters": a dict, "performance": a dict, "checkpoint": a str
                                 "whether_only_small_samples_are_used_for_eval": a bool
                             }
        Returns:
            None
        """
        from backend.src.basic_tools.data_structure.constant import Directories
        from backend.src.basic_tools.data_loader import JSONSaver
        from backend.src.basic_tools.misc import TimeOperators
        summary_file_dir_header = Directories.HuggingFacePath().evaluation_summary_file_header
        time_now = TimeOperators.get_current_time_in_vanilla_format_where_white_space_is_removed()
        summary_file_name = f"{self.log_dir_root}/{summary_file_dir_header}_at_{time_now}"
        print(f"[SAVING] evaluation summary at {summary_file_name}")
        JSONSaver(dir_=summary_file_name, data_=summary_results)()

    def get_summary_results(self):
        """
        Returns:
            a list of dict where each dict has the format of
            {
                "parameters": a dict, "performance": a dict, "checkpoint": a str
                "whether_only_small_samples_are_used_for_eval": a bool
            }
        """
        from backend.src.basic_tools.data_loader import JSONLoader
        mapping = self.get_mapping_from_param_info_file_path_to_performance_file_path()
        basket_list = []
        for param_info_file, performance_file in mapping.get_flattened_key_value_list():
            param_info = JSONLoader(dir_=param_info_file)()
            performance = JSONLoader(dir_=performance_file)()
            basket = {
                "parameters": param_info, "performance": performance,
                "checkpoint": self.grep_check_point_name_from_performance_file_path(
                    performance_file_path=performance_file
                ),
                "whether_only_small_samples_are_used_for_eval": (".small_sample" in performance_file)
            }
            basket_list.append(basket)
        basket_list = sorted(basket_list, key=lambda x: float(x['performance']['accuracy']), reverse=True)
        return basket_list

    @staticmethod
    def grep_check_point_name_from_performance_file_path(performance_file_path):
        """
        Args:
            performance_file_path: a str

        Returns:
            a str
        """
        name_ = performance_file_path.split("checkpoint")[-1].split(".")[0].replace("-","")
        return name_

    def get_mapping_from_param_info_file_path_to_performance_file_path(self):
        """
        Returns:
            a DictOfNonDuplicatedList that has the key of str and the value of the list of str
            the key is the parameter file path and the value is the list of performance file path
        """
        from glob import glob
        from backend.src.basic_tools.data_structure.dict_of_list import DictOfNonDuplicatedList
        to_return = DictOfNonDuplicatedList({})
        param_info_file_path_list = glob(self.log_dir_root + "/*.opt")
        performance_file_path_list = glob(self.log_dir_root + "/*.performance*")
        for performance_file_path in performance_file_path_list:
            matching_param_info_file_path =\
                self.get_param_info_file_matching_with_the_performance_file(
                    performance_file_path=performance_file_path, param_info_file_path_list=param_info_file_path_list
                )
            if not matching_param_info_file_path:
                continue
            to_return.add_new_key_value(new_key=matching_param_info_file_path, new_value=performance_file_path)
        return to_return

    @staticmethod
    def get_param_info_file_matching_with_the_performance_file(performance_file_path, param_info_file_path_list):
        """
        Args:
            performance_file_path: a str
            param_info_file_path_list: a list of str

        Returns:
            a str that ends with ".opt"
            or None (if there is no match)
        """
        file_name_key_list = [x.replace(".opt", "") for x in param_info_file_path_list]
        matching_file_name_list = [x for x in file_name_key_list if x in performance_file_path]
        if len(matching_file_name_list) == 1:
            to_return = matching_file_name_list[0] + ".opt"
        elif len(matching_file_name_list) == 0:
            to_return = None
        else:
            raise AssertionError("the matching files name should be found at most one time as caching is used")
        return to_return


class HuggingFaceModelEvaluationScheduler(HuggingFaceBaseClass):
    def __init__(self, gpu_id=0):
        """
        Args:
            gpu_id: an int
        """
        super().__init__(gpu_id=gpu_id)
        from backend.src.hugging_face_model_training_scheduler import HuggingFaceModelTrainingScheduler
        self.training_scheduler = HuggingFaceModelTrainingScheduler()

    def __call__(self, target_check_point_dir_list, use_small_sample, ignore_cache):
        """
        Args:
            target_check_point_dir_list: a list of str
            use_small_sample: a bool
            ignore_cache: a bool

        Returns:
            None
        """
        self.auto_eval(
            target_check_point_dir_list=target_check_point_dir_list,
            use_small_sample=use_small_sample, ignore_cache=ignore_cache
        )

    def auto_eval(self, target_check_point_dir_list, use_small_sample, ignore_cache):
        """
        Args:
            target_check_point_dir_list: a list of str
            use_small_sample: a bool
            ignore_cache: a bool

        Returns:
            None
        """
        from tqdm import tqdm
        from os import path
        from backend.src.basic_tools.data_loader import JSONSaver
        print(f"[EVALUATING] {len(target_check_point_dir_list)} checkpoints")
        for check_point_path in tqdm(target_check_point_dir_list):
            tokenizer = self.load_tokenizer_from_model_checkpoint_dir(checkpoint_path=check_point_path)
            _, test_data = self.training_scheduler.get_train_and_test_data(
                use_original_data_format=True, tokenizer=tokenizer
            )
            check_point_name = "_".join(check_point_path.split("/")[-2:])
            performance_log_dir = f"{self.log_dir_root}/{check_point_name}.performance_log"
            if use_small_sample:
                performance_log_dir += ".small_sample"
            if ignore_cache:
                pass
            elif path.isfile(performance_log_dir):
                print(f"[Found Cache] skipping {performance_log_dir}")
                continue
            performance = self.eval_model(
                test_data=test_data, checkpoint_path=check_point_path,
                use_small_sample=use_small_sample
            )
            performance['model_path'] = check_point_path
            performance['test_results_on_explicitly_given_dataset'] = self.eval_model(
                test_data=self.load_explicitly_given_dataset(), checkpoint_path=check_point_path,
                use_small_sample=use_small_sample
            )
            print(f"[Saving] {performance_log_dir}")
            JSONSaver(dir_=performance_log_dir, data_=performance)()

    def eval_model(self, test_data, checkpoint_path, use_small_sample):
        """
        Args:
            test_data: a Dataset or a list of dict where each dict has the key of `laws_service_id` and `fact`
            checkpoint_path: a str
            use_small_sample: a bool

        Returns:
            a dict that has a format of {"accuracy": a float, "f1": a float}
        """
        labels, predicted_labels = self.get_labels_and_predictions(
            test_data=test_data, checkpoint_path=checkpoint_path, use_small_sample=use_small_sample
        )
        accuracy_and_f1_dict = self.training_scheduler.get_accuracy_and_f1_for_predictions_and_labels(
            predictions=predicted_labels, labels=labels
        )
        return accuracy_and_f1_dict
    
    def get_labels_and_predictions(self, test_data, checkpoint_path, use_small_sample):
        """
        Args:
            test_data: a Dataset or a list of dict where each dict has the key of `laws_service_id` and `fact`
            checkpoint_path: a str
            use_small_sample: a bool

        Returns:
            a tuple of (list of str, list of str), which is (labels, predicted_labels)
            e.g., (["2", "1"], ["2", "5"])
        """
        from tqdm import tqdm
        tokenizer = self.load_tokenizer_from_model_checkpoint_dir(checkpoint_path=checkpoint_path)
        facts, labels, pipe = self.get_formatted_facts_labels_and_pipe_from_test_data_and_model_dir(
            test_data=test_data, model_dir=checkpoint_path, tokenizer=tokenizer, use_small_sample=use_small_sample
        )
        predictions = []
        print('[Processing] Comparing Model Predictions & Correct Labels')
        for fact_str in tqdm(facts):
            prediction = pipe(fact_str)[0]
            predictions.append(prediction)
        predicted_labels = [basket['label'].replace("LABEL_", "") for basket in predictions]
        return labels, predicted_labels

    @staticmethod
    def load_tokenizer_from_model_checkpoint_dir(checkpoint_path):
        """
        Args:
            checkpoint_path: a str

        Returns:
            a Tokenizer
        """
        from transformers import AutoTokenizer
        from backend.src.basic_tools.data_loader import JSONLoader
        from backend.src.basic_tools.data_structure.constant import Directories
        model_dir = "/".join(checkpoint_path.split("/")[:-1])
        hf_path = Directories.HuggingFacePath()
        model_file_name = model_dir.split("/")[-1]
        opt_file_dir = f"{hf_path.log_dir_root}/{model_file_name}.opt"
        basket_from_opt_file = JSONLoader(dir_=opt_file_dir)()
        hf_model_name = basket_from_opt_file['hugging_face_model_name']
        tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
        return tokenizer

    def get_formatted_facts_labels_and_pipe_from_test_data_and_model_dir(self, test_data, model_dir,
                                                                         tokenizer, use_small_sample):
        """
        Args:
            test_data: a Dataset or a list of dict where each dict has the key of `laws_service_id` and `fact`
            model_dir: a str
            tokenizer: a Tokenizer
            use_small_sample: a bool

        Returns:
            a tuple of (list of str, list of str, pipe)
        """
        from transformers import AutoModelForSequenceClassification, TextClassificationPipeline
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer)
        formatted_facts, formatted_labels = self.get_formatted_facts_and_labels_from_test_data(
            test_data=test_data, tokenizer=tokenizer, use_small_sample=use_small_sample
        )
        return formatted_facts, formatted_labels, pipe

    @staticmethod
    def get_formatted_facts_and_labels_from_test_data(test_data, tokenizer, use_small_sample):
        """
        Args:
            test_data: a Dataset or a list of dict where each dict has the key of `laws_service_id` and `fact`
            tokenizer: a Tokenizer
            use_small_sample: a bool

        Returns:
            a tuple of (list of str, list of str)
        """
        from backend.src.basic_tools.data_structure.constant import Numbers
        formatted_labels = [str(x['laws_service_id']) for x in test_data]
        facts_from_test_data = [x['fact'] for x in test_data]
        if use_small_sample:
            formatted_labels = formatted_labels[:Numbers.A_MODERATELY_SMALL_NUMBER]
            facts_from_test_data = facts_from_test_data[:Numbers.A_MODERATELY_SMALL_NUMBER]
        facts_tokenized_objs = [tokenizer(
            x, padding='max_length', truncation='longest_first', max_length=512-2
        ) for x in facts_from_test_data]
        facts_tokens_from_tokenized = [tokenizer.convert_ids_to_tokens(x.input_ids)
                                       for x in facts_tokenized_objs]
        formatted_facts = [tokenizer.convert_tokens_to_string(x)
                           for x in facts_tokens_from_tokenized]
        return formatted_facts, formatted_labels

    @staticmethod
    def load_explicitly_given_dataset():
        """
        Returns:
            a list of dict where each dict has the format of
            {"fact": a str, "laws_service_id": an int, ...}
        """
        from backend.src.basic_tools.data_loader import JSONLoader
        from backend.src.basic_tools.data_structure.constant import Directories
        loaded_ = JSONLoader(dir_=Directories.HuggingFacePath().explicitly_given_test_data)()
        for basket in loaded_:
            assert "fact" in basket
            assert "laws_service_id" in basket
        return loaded_
