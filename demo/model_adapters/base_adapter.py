from abc import ABC, abstractmethod


class BaseModelAdapter(ABC):
    @abstractmethod
    def load_model(self, model_path, processed_data_path=None):
        raise NotImplementedError

    @abstractmethod
    def train_model(self, train_data_path, base_checkpoint=None, model_version=None):
        raise NotImplementedError

    @abstractmethod
    def predict_period(self, model_path, processed_data_path, output_path, model_version=None):
        raise NotImplementedError

    @abstractmethod
    def evaluate_model(self, model_path, processed_data_path, actual_path=None, model_version=None):
        raise NotImplementedError
