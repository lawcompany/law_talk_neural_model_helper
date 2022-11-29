import random


class TimeOperators:
    def __init__(self):
        pass

    @staticmethod
    def get_current_time_in_vanilla_format_where_white_space_is_removed():
        """ Returns: a str """
        from datetime import datetime
        time_now = str(datetime.now())
        time_now = time_now.replace("/", "_").replace(" ", "_")
        return time_now


class CudaOperators:
    def __init__(self):
        pass

    @staticmethod
    def load_env_params():
        """ Returns: a dict """
        from utils.data_loader import JSONLoader
        from utils.constant import Directories
        env_params = JSONLoader(dir_=Directories.ENV_PARAM_DIR)()
        return env_params

    @staticmethod
    def get_the_num_of_available_gpus():
        """ Returns: an int """
        from torch.cuda import is_available, device_count
        if is_available():
            num_gpu = device_count()
        else:
            num_gpu = 0
        return num_gpu


class ListOperators:
    def __init__(self):
        pass

    @staticmethod
    def rm_dup(list_):
        """
        Args:
            list_: a list of obj that contains `equal` op.

        Returns:
            a list of the objs each of which has the same type
        """
        return list(set(list_))

    @staticmethod
    def split_large_list_into_list_of_lists(long_list_to_split, len_sub_list=100):
        """
        Args:
            long_list_to_split: a list
            len_sub_list: an int

        Returns:
            a list of list
        """
        list_of_sub_list = [long_list_to_split[x:x + len_sub_list]
                            for x in range(0, len(long_list_to_split), len_sub_list)]
        return list_of_sub_list

    @staticmethod
    def identity(obj):
        return obj

    @staticmethod
    def convert_basket_list_into_a_dict(basket_list, target_key):
        """
        Args:
            basket_list: a list of dict
            target_key: a str, which would be the key of the baskets within the basket list
                        where the value according to the key would be a new key for the basket in the new dict.

        Returns:
            a dict
        """
        dict_to_return = dict()
        for basket in basket_list:
            if target_key in basket:
                pass
            else:
                raise RuntimeError
            dict_to_return[target_key] = basket
        return dict_to_return

    def get_phrase_parsing_results_based_on_start_and_end_tags(self, token_list, start_tag, end_tag):
        """
        This function returns the index set of the targets of the parsing
        based on the given start tag and the given end tag.
        The indexing of the targets come with its unique id, which is
        the order that it occurs, starting from the front side of the sentence.

        Args:
            token_list: a list of str
            start_tag: a str
            end_tag: a str

        Returns:
            a UnidirectionalOneToOneMapping from a int to a list
            where the int is an index that starts from 0, indicating
            the phrase identified by the start tag and the end tag.
            The list is the list of the index that corresponds to the tokens in the token_list
        """
        assert self._is_token_list_have_the_same_number_of_start_and_end_tags(token_list, start_tag, end_tag)
        phrase_index = 0
        _is_in_the_phrase = False
        token_indices_in_phrase = []
        map_ = UnidirectionalOneToOneMapping({})
        for index, token in enumerate(token_list):
            if token == start_tag:
                _is_in_the_phrase = True
            elif token == end_tag:
                map_.add_new_pair(phrase_index, token_indices_in_phrase)
                _is_in_the_phrase = False
                phrase_index += 1
                token_indices_in_phrase = []
            else:
                if _is_in_the_phrase:
                    token_indices_in_phrase.append(index)
                else:
                    pass
        return map_

    @staticmethod
    def _is_token_list_have_the_same_number_of_start_and_end_tags(token_list, start_tag, end_tag):
        return token_list.count(start_tag) == token_list.count(end_tag)

    def remove_surface_token_duplication(self, list_, function_getting_surface=None):
        if function_getting_surface:
            pass
        else:
            function_getting_surface = self.identity
        rev_list = []
        surface_list = []
        for item in list_:
            surface_ = function_getting_surface(item)
            if surface_ in surface_list:
                pass
            else:
                surface_list.append(surface_)
                rev_list.append(item)
        return rev_list

    def has_no_surface_token_duplication(self, list_, function_getting_surface=None):
        rev_list = self.remove_surface_token_duplication(list_, function_getting_surface)
        return list_ == rev_list

    def get_duplicated_items(self, list_, function_getting_surface=None):
        if function_getting_surface:
            pass
        else:
            function_getting_surface = self.identity
        duplicated_items = []
        surface_list = []
        for item in list_:
            surface_ = function_getting_surface(item)
            if surface_ in surface_list:
                duplicated_items.append(item)
            else:
                surface_list.append(surface_)
        return duplicated_items

    def get_all_n_gram_with_varying_window_size(self, list_, min_window_size=1, max_window_size=None):
        if max_window_size:
            pass
        else:
            max_window_size = len(list_)
        window_size_list = list(range(min_window_size, max_window_size + 1))
        n_gram_list = []
        for window_size in window_size_list:
            n_gram_list += self.get_n_gram(list_, window_size)
        return n_gram_list

    def get_alignment_in_unidirectional_mapping_form(self, from_=None, to_=None):
        alignment_map_predicted = self.exhaustive_alignment_bidirectional(
            from_, to_
        )
        alignment_mapping = UnidirectionalOneToOneMapping({})
        for span_tag_idx, input_idx in alignment_map_predicted:
            alignment_mapping.add_new_pair(span_tag_idx, input_idx)
        return alignment_mapping

    def exhaustive_alignment(self, list_1, list_2):
        if len(list_1) >= len(list_2):
            matching_idx_tuple = self.unidirectional_exhaustive_alignment(list_1, list_2)
        else:
            matching_idx_tuple = self.unidirectional_exhaustive_alignment(list_2, list_1)
            matching_idx_tuple = [(y, x) for (x, y) in matching_idx_tuple]
        return matching_idx_tuple

    def unidirectional_exhaustive_alignment(self, list_1, list_2):
        max_len = -1
        max_item = []
        for idx, _ in enumerate(list_1):
            list_1_part = list_1[idx:]
            matching_idx_tuple = self.lazy_alignment(list_1_part, list_2)
            if len(matching_idx_tuple) >= max_len:
                for jdx, tuple_ in enumerate(matching_idx_tuple):
                    x, y = tuple_
                    matching_idx_tuple[jdx] = [x + idx, y]
                max_item = matching_idx_tuple
                max_len = len(matching_idx_tuple)
        matching_idx_tuple = max_item
        return matching_idx_tuple

    def exhaustive_alignment_bidirectional(self, list_1, list_2):
        forward_matching = self.exhaustive_alignment(list_1, list_2)
        reversed_list_1 = list(reversed(list_1))
        reversed_list_2 = list(reversed(list_2))
        backward_matching = self.exhaustive_alignment(reversed_list_1, reversed_list_2)
        if len(forward_matching) >= len(backward_matching):
            return forward_matching
        else:
            backward_matching = [(len(list_1) - x - 1,
                                  len(list_2) - y - 1)
                                 for (x, y) in backward_matching]
            return backward_matching

    @staticmethod
    def get_second_order_iterator(list_1, list_2):
        tuple_list = []
        for item_1 in list_1:
            for item_2 in list_2:
                tuple_list.append((item_1, item_2))
        return tuple_list

    @staticmethod
    def flatten_list_of_list_into_list(list_of_list):
        return [item for sublist in list_of_list for item in sublist]

    @staticmethod
    def get_n_gram(list_, window_size):
        """
        Args:
            list_: a list that contains the tokens that is the target of the n-gram extraction
            window_size: an integer, which is the "n".

        Returns:
            a list of list, where each list is the n-gram of the tokens
        """
        n_gram_list = []
        for idx, _ in enumerate(list_):
            n_gram = list_[idx:idx + window_size]
            if len(n_gram) == window_size:
                n_gram_list.append(n_gram)
            else:
                pass
        return n_gram_list

    @staticmethod
    def get_all_combinations_with_varying_length(list_, min_len=1, max_len=None):
        """
        For example if the list_ = [a, b, c, d]
        and the min_len = 2, and max_len = 3
        then the return value will be [[a, b],[a, c],[a, d],...[a, b, c],[b, c, d],...]

        Args:
            list_: the list of items of which we want to make combinations
            min_len: the min len of the combination.
            max_len: the max len of the combination.

        Returns:
            the list of list of item, like the example.
        """
        from itertools import combinations
        if max_len:
            pass
        else:
            max_len = len(list_)
        len_list = list(range(min_len, max_len + 1))
        combination_list = []
        for len_ in len_list:
            combination_list += combinations(list_, len_)
        return combination_list

    @staticmethod
    def lazy_alignment(list_1, list_2):
        matching_idx_tuple = []
        pos1 = -1
        pos2 = -1
        for idx1, token1 in enumerate(list_1):
            if idx1 <= pos1:
                continue
            for idx2, token2 in enumerate(list_2):
                if idx2 <= pos2:
                    continue
                if token1 == token2:
                    pos1 = idx1
                    pos2 = idx2
                    matching_idx_tuple.append([idx1, idx2])
                    break
        return matching_idx_tuple


class PlainTextOperators:
    def __init__(self):
        from backend.src.basic_tools.data_structure.constant import Tokens, Directories
        import string
        from nltk.corpus import stopwords
        self.english_stopwords = stopwords.words('english')
        # self.korean_stop_words_root_dir = Directories.KoreanDataPaths
        # self.korean_stop_words = self.load_korean_stop_words()
        self.english_alphabets = list(string.ascii_letters)
        self.delimiter = "[DELIMITER]"
        self.special_tokens = Tokens.SPECIAL_TOKENS
        # self.explicit_vocab_dict_dir = Directories.EXPLICIT_VOCAB_DICT
        # self.wordnet_lemmatizer = None
        # self.stanford_nlp = None

    @staticmethod
    def convert_datetime_str_into_seconds(time_):
        """
        Args:
            time_: a str with format such as 2022-07-01 00:02:25

        Returns:
            an int
        """
        from datetime import datetime
        time_obj = datetime.strptime(time_, '%Y-%m-%d %H:%M:%S')
        time_since_1990 = time_obj - datetime(1990, 1, 1)
        time_since_1990_in_seconds = int(time_since_1990.total_seconds())
        return time_since_1990_in_seconds

    @staticmethod
    def get_legal_safe_keywords(whether_to_use_kb_to_extend_safe_keywords=True, extension_depth=3):
        """
        Args:
            whether_to_use_kb_to_extend_safe_keywords: a bool
            extension_depth: an int

        Returns:
            a list of str
        """
        from backend.src.basic_tools.data_structure.constant import Tokens
        from backend.src.knowledge_base_handler import KnowledgeBaseLoader
        list_op = ListOperators()
        knowledge_base = KnowledgeBaseLoader()()
        safe_keywords = Tokens.SAFE_KEYWORDS
        if whether_to_use_kb_to_extend_safe_keywords:
            pass
        else:
            return safe_keywords
        for _ in range(extension_depth):
            rev_safe_keywords = [knowledge_base[x] for x in safe_keywords if x in knowledge_base]
            rev_safe_keywords = list_op.flatten_list_of_list_into_list(rev_safe_keywords) + safe_keywords
            safe_keywords = list(set(rev_safe_keywords))
        safe_keywords = sorted(safe_keywords)
        print(safe_keywords)
        return safe_keywords

    @staticmethod
    def pycharm_debugger_unicode_decode_error_handler(obj):
        """
        Args:
            obj: untyped

        Returns:
            the same type with obj
        """
        def replace_issued_token(txt):
            return txt.replace("ㆍ", " ")
        if isinstance(obj, dict):
            to_return = dict()
            for key_, val_ in obj.items():
                rev_key, rev_val = replace_issued_token(key_), replace_issued_token(val_)
                to_return[rev_key] = rev_val
        elif isinstance(obj, list):
            to_return = [replace_issued_token(x) for x in obj]
        elif isinstance(obj, str):
            to_return = replace_issued_token(obj)
        else:
            raise TypeError
        return to_return

    def filter_out_text_spans_that_are_overlapped_to_others_in_given_span_lists(self, given_span_list,
                                                                                overlap_min_limit=1):
        """
        Args:
            given_span_list: a list of str
            overlap_min_limit: an int

        Returns:
            a list of str
        """
        already_included = set()
        for span_ in given_span_list:
            list_of_overlap_words_len_to_the_ones_already_included =\
                [len(self.get_word_overlaps_between_two_spans(span_, x)) for x in already_included]
            min_overlap_len = min(list_of_overlap_words_len_to_the_ones_already_included)
            if min_overlap_len > overlap_min_limit:
                pass
            else:
                already_included.add(span_)
        return sorted(list(already_included))

    @staticmethod
    def get_word_overlaps_between_two_spans(pivot_span_to_compare_with, span_to_compare):
        """
        Args:
            span_to_compare: a str
            pivot_span_to_compare_with: a str

        Returns:
            a list of str, which is the overlapped words
        """
        from backend.src.tokenizer import SimpleTokenizer
        tokenizer = SimpleTokenizer()
        words = tokenizer(text=span_to_compare)
        words_in_pivot = set(tokenizer(text=pivot_span_to_compare_with))
        overlapped_words = []
        for word in words:
            if word in words_in_pivot:
                overlapped_words.append(word)
            else:
                pass
        return overlapped_words

    @staticmethod
    def find_all_occurrence_indices_of_a_word_within_the_given_text(word, given_text, loop_max=100):
        """
        Args:
            word: a str
            given_text: a str
            loop_max: an int

        Returns:
            a list of int, which is the begin_offsets of the words within the given text
            where the offset is measured in the manner of 'character-level' offset.
        """
        start = 0
        loop_ = 0
        while True:
            loop_ += 1
            start = given_text.find(word, start)
            if loop_ > loop_max:
                return
            elif start == -1:
                return
            else:
                pass
            yield start
            start += len(word)

    def load_korean_stop_words(self):
        """ Returns: None """
        from glob import glob
        from backend.src.basic_tools.data_loader import TEXTLoader
        txt_files = glob(self.korean_stop_words_root_dir + "/*.txt")
        stop_words = []
        for txt_file in txt_files:
            lines = TEXTLoader(dir_=txt_file)().split("\n")
            words = [x for x in lines if x]
            stop_words += words
        return stop_words

    def remove_korean_stop_words(self, token_list):
        """
        Args:
            token_list: a list of str

        Returns:
            a list of str
        """
        if self.korean_stop_words:
            pass
        else:
            self.load_korean_stop_words()
        rev_token_list = []
        for token in token_list:
            if token in self.korean_stop_words:
                pass
            else:
                rev_token_list.append(token)
        return rev_token_list

    def remove_tokens_that_include_english_alphabet_from_given_token_list(self, token_list):
        """
        Args:
            token_list: a list of str

        Returns:
            a list of str
        """
        rev_tokens = []
        for token in token_list:
            whether_ = self.whether_given_string_contains_english_alphabet(token)
            if whether_:
                pass
            else:
                rev_tokens.append(token)
        return rev_tokens

    def whether_given_string_contains_english_alphabet(self, str_):
        """
        Args:
            str_: a str

        Returns:
            a bool
        """
        for alphabet in self.english_alphabets:
            if alphabet in str_:
                return True
            else:
                pass
        return False

    def get_most_frequent_words_from_given_documents(self, documents, top_k):
        """
        Args:
            documents: a list of str
            top_k: an int

        Returns:
            a list of str, where each str is a word and the len(list) is top_k
        """
        word2count = self.get_word2count_from_given_documents(documents=documents)
        word_list = sorted(list(word2count.keys()), key=lambda x: word2count[x], reverse=True)
        if len(word_list) > top_k:
            word_list = word_list[:top_k]
        else:
            pass
        return word_list

    @staticmethod
    def get_word2count_from_given_documents(documents):
        """
        It should be noted that each word is split by ' '
        Args:
            documents: a list of str

        Returns:
            a dict where the key is a str and the value is an int
            where the key is the word and the value is the count of the words within the documents
        """
        from ..basic_tools.data_structure.dict_of_list import DictOfList
        from .data_structure.constant import Tokens
        stop_words = set(Tokens.KOR_STOP_WORDS + Tokens.KOR_SUFFIX)
        word2count, word2collection = dict(), DictOfList({})
        word_list = []
        for document in documents:
            word_list += document.split(" ")
        word_list = [x for x in word_list if x]
        for word in word_list:
            if word in stop_words:
                continue
            else:
                pass
            word2collection.add_new_key_value(new_key=word, new_value=1)
        for word, val_list in word2collection.get_key_val_iterator():
            word2count[word] = sum(val_list)
        return word2count

    @staticmethod
    def split_given_string_with_multiple_delimiters(delimiter_list, given_str, tokenizer='simple'):
        """
        Args:
            delimiter_list: a list of str
            given_str: a str
            tokenizer: the name for tokenizer such as 'simple' or 'stanford_core_nlp'

        Returns:
            a list of str
        """
        from backend.src.tokenizer import SimpleTokenizer
        if tokenizer == 'simple':
            tokenizer = SimpleTokenizer()
        else:
            raise AssertionError("now, only the simple tokenizer is available")
        tokenized = tokenizer(text=given_str)
        delimiter_index_list = []
        for idx, token in enumerate(tokenized):
            if token in delimiter_list:
                delimiter_index_list.append(idx)
            else:
                pass
        to_return = []
        cache = []
        for idx, token in enumerate(tokenized):
            if idx in delimiter_index_list:
                to_return.append(" ".join(cache))
                to_return.append(token)
                cache = []
            else:
                cache.append(token)
        if cache:
            to_return.append(" ".join(cache))
        else:
            pass
        to_return = [x for x in to_return if x]
        return to_return

    @staticmethod
    def split_string_list_with_multiple_delimiters(delimiter_list, lines):
        """
        Args:
            delimiter_list: a list of str
            lines: a list of str

        Returns:
            a list of the list of str, where the nested list is the lines sectioned by one of the delimiters.
        """
        def whether_line_contains_the_delimiter(delimiter_list_, line_):
            whether_ = False
            for delimiter_ in delimiter_list_:
                if delimiter_ in line_:
                    whether_ = True
                else:
                    pass
            return whether_
        delimiter_idx_list = []
        for idx, line in enumerate(lines):
            if whether_line_contains_the_delimiter(delimiter_list_=delimiter_list, line_=line):
                delimiter_idx_list.append(idx)
            else:
                pass
        to_return = []
        for meta_idx, idx in enumerate(delimiter_idx_list):
            if meta_idx == 0:
                continue
            else:
                pass
            previous_idx = delimiter_idx_list[meta_idx-1]
            to_return.append(lines[previous_idx+1:idx])
        return to_return

    @staticmethod
    def naive_split(text):
        """
        Args:
            text: a str

        Returns:
            a list of str
        """
        text = text.replace(".", " ").replace(",", " ").replace("\n", " ")
        token_list = text.split(" ")
        token_list = [x for x in token_list if x]
        return token_list

    def add_white_space_to_the_left_of_a_special_character(self, text, exceptional_special_token_list=None):
        if exceptional_special_token_list is None:
            exceptional_special_token_list = []
        for special_token in self.special_tokens:
            if special_token in exceptional_special_token_list:
                continue
            text = text.replace(special_token, f" {special_token}")
            text = text.replace(f"  {special_token}", f" {special_token}")
        return text

    def get_token_pos_list_converted_to_base_form(self, token_pos_list):
        """
        Args:
            token_pos_list: list of tuple (token, pos_tag)

        Returns:
            a list of str
        """
        return [self.converter_for_equality(token, tag)
                for (token, tag) in token_pos_list]

    def get_token_list_converted_to_base_form(self, token_list):
        """
        Args:
            token_list: a list of str

        Returns:
            a list of str
        """
        return [self.converter_for_equality(token)
                for token in token_list]

    def converter_for_equality(self, word, tag=None):
        from .data_loader import JSONLoader
        explicit_dict = JSONLoader(self.explicit_vocab_dict_dir)()
        if word in explicit_dict:
            return explicit_dict[word]
        else:
            return self.get_base_form_using_wordnet_lemmatizer(word, tag)

    def get_base_form_using_wordnet_lemmatizer(self, word, tag):
        from nltk.stem import WordNetLemmatizer
        if self.wordnet_lemmatizer:
            pass
        else:
            self.wordnet_lemmatizer = WordNetLemmatizer()
        if tag:
            first_two_letters_of_tag = tag[:2]
            if first_two_letters_of_tag == "VB":
                pos = "v"
            elif first_two_letters_of_tag == "NN":
                pos = "n"
            elif first_two_letters_of_tag == "RB":
                pos = "r"
            elif first_two_letters_of_tag == "JJ":
                pos = "a"
            else:
                pos = "n"
            return self.wordnet_lemmatizer.lemmatize(word, pos=pos).lower()
        else:
            return self.wordnet_lemmatizer.lemmatize(word).lower()

    @staticmethod
    def get_character_difference(text_1, text_2):
        """
        Args:
            text_1: a str
            text_2: a str

        Returns:
            a tuple, (str, str), which is the difference between the two text snippets
        """
        for idx, _ in enumerate(text_1):
            to_compare_1 = text_1[:idx + 1]
            if len(text_2) <= idx + 1:
                return text_1[idx + 1:], ""
            else:
                to_compare_2 = text_2[:idx + 1]
            if to_compare_1 == to_compare_2:
                pass
            else:
                return text_1[idx + 1:], text_2[idx + 1:]
        return "", ""

    @staticmethod
    def is_containing_all_the_given_token_list(str_, token_list):
        """ Returns: if all the tokens are within the str_, then True else False """
        for token in token_list:
            if token in str_:
                pass
            else:
                return False
        return True

    def is_containing_special_character(self, str_):
        for token_ in self.special_tokens:
            if token_ in str_:
                return True
        return False

    def remove_special_tokens(self, str_, special_tokens=None):
        if special_tokens:
            pass
        else:
            special_tokens = self.special_tokens
        for token_ in special_tokens:
            str_ = str_.replace(token_, "")
        return str_.strip()

    def replace_special_tokens(self, str_, special_tokens=None):
        if special_tokens:
            pass
        else:
            special_tokens = self.special_tokens
        for token_ in special_tokens:
            str_ = str_.replace(token_, " ")
        return str_.strip().replace("  ", " ")

    @staticmethod
    def remove_non_utf_8_characters(str_):
        rev_ = str_.decode('utf-8', 'ignore').encode("utf-8")
        return rev_

    @staticmethod
    def remove_digits(str_):
        rev_str = ''.join([i for i in str_ if not i.isdigit()])
        return rev_str

    def concatenate(self, list_):
        """
        Args:
            list_: the list that is given as an argument.

        Returns:
            the concatenated text
        """
        if not list_:
            concatenated = ""
            return concatenated
        start, middle = list_[0], list_[1:]
        concatenated = str(start)
        for item_ in middle:
            concatenated += self.delimiter
            concatenated += str(item_)
        return concatenated


class PathFinder:
    def __init__(self, graph, start, end):
        self.graph = graph
        self.start = start
        self.end = end

    def get_path_list(self):
        return self._find_all_paths(self.graph, self.start, self.end)

    def get_shortest_path(self):
        return self._find_shortest_path(self.graph, self.start, self.end)

    def _find_shortest_path(self, graph, start, end, path=None):
        # code from https://www.python.org/doc/essays/graphs/
        if path is None:
            path = []
        path = path + [start]
        if start == end:
            return path
        if start not in graph:
            return None
        shortest = None
        for node in graph[start]:
            if node not in path:
                newpath = self._find_shortest_path(graph, node, end, path)
                if newpath:
                    if not shortest or len(newpath) < len(shortest):
                        shortest = newpath
        return shortest

    def _find_all_paths(self, graph, start, end, path=None):
        # code from https://www.python.org/doc/essays/graphs/
        if path is None:
            path = []
        path = path + [start]
        if start == end:
            return [path]
        if start not in graph:
            return []
        paths = []
        for node in graph[start]:
            if node not in path:
                newpaths = self._find_all_paths(graph, node, end, path)
                for newpath in newpaths:
                    paths.append(newpath)
        return paths

    def _find_path(self, graph, start, end, path=None):
        # code from https://www.python.org/doc/essays/graphs/
        if path is None:
            path = []
        path = path + [start]
        if start == end:
            return path
        if start not in graph:
            return None
        for node in graph[start]:
            if node not in path:
                newpath = self._find_path(graph, node, end, path)
                if newpath:
                    return newpath
        return None


class MethodRunner:
    def run_all_methods_in_class(self, class_):
        """
        Args:
            class_: the target class, within which we want to call all the functions

        Returns:
            None
        """
        methods = self._get_all_methods_in_class(class_)
        for method in methods:
            method()

    @staticmethod
    def _get_all_methods_in_class(class_):
        method_names = [method for method in dir(class_) if callable(getattr(class_, method)) if
                        not method.startswith('_')]
        methods = [getattr(class_, method_name) for method_name in method_names]
        return methods
