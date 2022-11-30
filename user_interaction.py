class HumanInteractionHelper:
    def __init__(self, model_checkpoint_dir, gpu_id):
        from eval_models import HuggingFaceModelEvaluationScheduler
        self.model_checkpoint_dir = model_checkpoint_dir
        self.evaluator = HuggingFaceModelEvaluationScheduler(gpu_id=gpu_id)

    def hugging_face_human_interaction(self, use_inputs_from_given_file):
        from utils.constant import Directories
        from utils.data_loader import JSONLoader
        hf_path = Directories.HuggingFacePath()
        if use_inputs_from_given_file:
            print(f"[Loading User Inputs] from file {hf_path.human_inputs_file_dir}")
            user_inputs = JSONLoader(dir_=hf_path.human_inputs_file_dir)()
            user_inputs = [x['input'] for x in user_inputs if 'input' in x]
        else:
            user_input = input("Enter input for the neural model:")
            print("[Received] input: " + user_input)
            user_inputs = [user_input]
        user_inputs_formatted = self.format_user_inputs_with_dummy_labels(user_inputs=user_inputs)
        _, predicted_labels = self.evaluator.get_labels_and_predictions(
            test_data=user_inputs_formatted, checkpoint_path=self.model_checkpoint_dir, use_small_sample=False
        )
        map_ = JSONLoader(dir_=Directories.LAW_SERVICE_MAPPING_DIR)()
        predictions_in_plain_text = [map_[str(x)] for x in predicted_labels]
        self.pretty_print_for_the_model_predictions(
            inputs=user_inputs, predictions=predictions_in_plain_text
        )

    @staticmethod
    def pretty_print_for_the_model_predictions(inputs, predictions):
        """
        Args:
            inputs: a list of str
            predictions: a list of str
        """
        from pprint import pprint
        assert len(inputs) == len(predictions)
        for idx, input_ in enumerate(inputs):
            prediction_ = predictions[idx]
            print(f"[USER INTERACTION] for the user input No. {idx+1} is as follows:")
            pprint({
                "[USER INPUT]": input_, "[MODEL PREDICTION]": prediction_
            })
            print("-"*30)

    @staticmethod
    def format_user_inputs_with_dummy_labels(user_inputs):
        """
        Args:
            user_inputs: a list of str

        Returns:
            a list of dict where each dict has the key of `laws_service_id` and `fact`
            where the value for `laws_service_id` is an int and that for `fact` is a str
        """
        to_return = []
        for user_input in user_inputs:
            to_return.append({"fact": user_input, "laws_service_id": 1})
        return to_return


class Interactor:
    def __init__(self):
        self.use_inputs_from_given_file = True

        from utils.misc import CudaOperators
        env_params = CudaOperators.load_env_params()
        self.gpu_id_to_use = env_params["gpu_id_to_use"]
        self.use_inputs_from_given_file = env_params["use_file_input_for_human_interaction"]

    def interact_with_user(self):
        model_check_point_dir = self._get_best_model_if_there_is_any_and_get_random_model_else()
        helper = HumanInteractionHelper(
            model_checkpoint_dir=model_check_point_dir, gpu_id=self.gpu_id_to_use,
        )
        helper.hugging_face_human_interaction(
            use_inputs_from_given_file=self.use_inputs_from_given_file
        )

    @staticmethod
    def _get_best_model_if_there_is_any_and_get_random_model_else():
        """ Returns: a str (a dir) """
        from glob import glob
        from eval_models import Evaluator
        from utils.constant import Directories
        from utils.misc import ListOperators
        from utils.data_loader import JSONLoader
        check_point_dir_list = Evaluator().get_check_point_dir_list_for_evaluation()
        hf_path = Directories.HuggingFacePath()
        summary_files = glob(f"{hf_path.log_dir_root}/{hf_path.evaluation_summary_file_header}*")
        summaries = ListOperators().flatten_list_of_list_into_list([JSONLoader(dir_=x)() for x in summary_files])
        summaries = sorted(summaries, key=lambda x: float(x['performance']['accuracy']), reverse=True)
        check_point_dir_list_from_summaries = [x['performance']['model_path'] for x in summaries]
        check_point_dir_list_from_summaries = [x for x in check_point_dir_list_from_summaries
                                               if x in check_point_dir_list]
        if check_point_dir_list_from_summaries:
            to_return = check_point_dir_list_from_summaries[0]
        else:
            from random import shuffle
            shuffle(check_point_dir_list)
            to_return = check_point_dir_list[0]
        return to_return


if __name__ == "__main__":
    Interactor().interact_with_user()
