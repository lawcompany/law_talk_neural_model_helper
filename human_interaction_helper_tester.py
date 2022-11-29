class HumanInteractionHelperTester:
    def __init__(self):
        self.gpu_id_to_use = 1
        self.use_inputs_from_given_file = True

    def main(self):
        self._test_hugging_face_human_interaction()

    def _test_hugging_face_human_interaction(self):
        from backend.src.human_interaction_helper import HumanInteractionHelper
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
        from testers.hugging_face_model_evaluation_scheduler_tester import HuggingFaceModelEvaluationSchedulerTester
        from backend.src.basic_tools.data_structure.constant import Directories
        from backend.src.basic_tools.misc import ListOperators
        from backend.src.basic_tools.data_loader import JSONLoader
        check_point_dir_list = HuggingFaceModelEvaluationSchedulerTester().get_check_point_dir_list_for_evaluation()
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
