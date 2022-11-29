class DictOfList:
    def __init__(self, initial_value):
        self.dict_of_list = initial_value

    def __len__(self):
        return len(list(self.get_all_elements()))

    def __add__(self, other):
        assert type(self) == type(other)
        for key_, list_ in other.dict_of_list.items():
            self.add_new_key_and_list_of_value(new_key=key_, new_list_of_value=list_)
        return self

    def __call__(self):
        return self.dict_of_list

    def __getitem__(self, item):
        return self.dict_of_list[item]

    def __contains__(self, item):
        return item in self.dict_of_list

    def __deepcopy__(self, memodict={}):
        from copy import deepcopy
        return DictOfList(deepcopy(self()))

    def merge_with(self, other):
        """
        Args:
            other: another DictOfList

        Returns:
            a DictOfList
        """
        from copy import deepcopy
        assert type(self) == type(other)
        merged = deepcopy(self)
        for key_, val_list in other.get_key_val_iterator():
            merged.add_new_key_and_list_of_value(new_key=key_, new_list_of_value=val_list)
        return merged

    def __eq__(self, other):
        condition_ = (type(self) == type(other))
        condition_ = condition_ and (self.dict_of_list.keys() == other.dict_of_list.keys())
        for key_ in self.dict_of_list.keys():
            condition_ = condition_ and (self.dict_of_list[key_] == other.dict_of_list[key_])
        return condition_

    def get_key_val_iterator(self):
        return self.dict_of_list.items()

    def get_all_elements(self):
        """
        Returns: a set of all the elements, not allowing duplication
        """
        elements = set()
        for key_, list_ in self.dict_of_list.items():
            elements.add(key_)
            for item in list_:
                elements.add(item)
        return elements

    def get_sub_dict_of_list_under(self, initial_point_to_start):
        """
        Recursion - Starting from the root_of_sub_tree, add children to the Tree,
                    and do the same for the newly added nodes until there is no children left to add.

        Args:
            initial_point_to_start: the initial point to start diving into the dict of list.

        Returns:
            sub_tree in a dict_of_list format
        """
        sub_dict_of_nondup_list = DictOfNonDuplicatedList({})
        for parent, child in self.get_iteration_from_given_sub_root(initial_point_to_start):
            sub_dict_of_nondup_list.add_new_key_value(parent, child)
        return sub_dict_of_nondup_list

    def get_iteration_from_given_sub_root(self, given_sub_root):
        """
        Args:
            given_sub_root: a Node

        Returns: the list of tuple which is (parent, child),
                 where the order is preserved. Starting from the given_sub_root,
                 and then the element is ordered by
                 "the distance" (if allows to speak crudely) between
                 the given_sub_root-and the parent in the dict.
        """
        parent_child_ordered_list = []
        queue_of_parents_to_add = [given_sub_root]
        whole_dict_of_list = self()
        _loop_itered, _num_loop_limit = 0, 1000
        while queue_of_parents_to_add:
            parents_to_delete_after_a_loop = []
            for parent in queue_of_parents_to_add:
                parents_to_delete_after_a_loop.append(parent)
                if parent in whole_dict_of_list:
                    children = whole_dict_of_list[parent]
                    for child in children:
                        parent_child_ordered_list.append((parent, child))
                    queue_of_parents_to_add += children
                else:
                    continue
            for key_to_delete in parents_to_delete_after_a_loop:
                queue_of_parents_to_add.remove(key_to_delete)
            _loop_itered += 1
            if _loop_itered > _num_loop_limit:
                raise StopIteration("number of loop is too big, debuging required")
        return parent_child_ordered_list

    def get_flattened_key_value_list(self):
        """

        Returns: the list of the tuple (key, value) which can be viewed as (parent, child).
                 for example, for the dict_of_list {"1": ["2", "3"], "2": ["3", "4", "5"]},
                 the list of the tuple to return will be, [("1", "2"), ("1", "3"),
                 ("2", "3"), ("2", "4"), ("2", "5"),]

        """
        tuple_list = []
        for key_, list_ in self.dict_of_list.items():
            for val_ in list_:
                tuple_list.append((key_, val_))
        return tuple_list

    def add_new_key_and_list_of_value(self, new_key, new_list_of_value):
        """
        Args:
            new_key: key of the new pair
            new_list_of_value: list of the values correspond to the key

        Returns:
            None
        """
        for new_value in new_list_of_value:
            self.add_new_key_value(new_key, new_value)

    def add_new_key_value(self, new_key, new_value):
        """
        Args:
            new_key: key of the new pair
            new_value: value of the new pair

        Returns:
            None
        """
        if new_key in self.dict_of_list:
            self.dict_of_list[new_key].append(new_value)
        else:
            self.dict_of_list[new_key] = [new_value]

    def update(self, other):
        """
        Args:
            other: a DictOfList

        Returns:
            a DictOfList
        """
        for key_, val_ in other.get_flattened_key_value_list():
            self.add_new_key_value(key_, val_)


class DictOfNonDuplicatedList(DictOfList):
    def __init__(self, initial_value):
        super().__init__(initial_value)
        for key_, list_ in self.dict_of_list.items():
            self._test_has_no_duplicacy_in_list(list_)

    def __deepcopy__(self, memodict={}):
        from copy import deepcopy
        return DictOfNonDuplicatedList(deepcopy(self()))

    def merge_with(self, other):
        """
        Args:
            other: a DictOfNonDuplicatedList

        Returns:
            a DictOfNonDuplicatedList
        """
        from copy import deepcopy
        merged = deepcopy(self)
        for key_, val_ in other.get_flattened_key_value_list():
            merged.add_new_key_value(key_, val_)
        return merged

    def add_new_key_value(self, new_key, new_value):
        """
        Args:
            new_key: key of the new pair
            new_value: value of the new pair

        Returns:
            None
        """
        if new_key in self.dict_of_list:
            self.dict_of_list[new_key] = self._append_without_duplication(self.dict_of_list[new_key], new_value)
        else:
            self.dict_of_list[new_key] = [new_value]

    @staticmethod
    def _append_without_duplication(list_, val_):
        if val_ in list_:
            pass
        else:
            list_.append(val_)
        return list_

    @staticmethod
    def _test_has_no_duplicacy_in_list(list_to_check):
        """
        Args:
            list_to_check: a list

        Returns:
            None
        """
        for item in list_to_check:
            assert (list_to_check.count(item) == 1)
