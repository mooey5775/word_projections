from abc import ABCMeta, abstractmethod

class BaseModel(metaclass=ABCMeta):
    @abstractmethod
    def get_word_vec(self, word):
        pass

    def get_word_vec_list(self, word):
        return self.get_word_vec(word).tolist()
