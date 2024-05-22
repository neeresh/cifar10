from blackboxattacks.surrogate_models.cifar100_surrogate_models import TrainModel


class CIFAR10SurrogateModels:

    def __init__(self, models, file_path):
        self.models = models
        self.filepath = file_path

    def _modify_dict_keys(self, original_dict):
        temp_dict = {}

        for key_tuple in original_dict.keys():
            model_name, layer = key_tuple
            new_key = f"{model_name}{layer}"
            temp_dict[new_key] = original_dict[key_tuple]

        return temp_dict

    def get_surrogate_models(self):
        """
        Syntax to load desired models
        models = [
            ('vgg', 16), ('resnet', 18), ('resnet', 50), ('resnet', 101),
            ('regnetx', '_200MF'), ('regnety', '_400MF'), ('mobilenetv2', None),
            ('resnext29', '_2x64d'), ('resnext29', '_32x4d'), ('simpledla', None),
            ('densenet', 121), ('preactresnet', 18), ('dpn', 92), ('dla', None)
        ]

        filepath = <path_to_the_model_implementations>
        """

        print(f"Loading models from {self.filepath}")
        loaded_models = {}
        for model_name, layer in self.models:
            if layer is not None:
                model = TrainModel(model_name, layer, initialize_weights=False)
            else:
                model = TrainModel(model_name, initialize_weights=False)

            loaded_model = model.load_model(file_path=self.filepath)
            loaded_models[(model_name, layer)] = loaded_model

        loaded_models = self._modify_dict_keys(loaded_models)

        return loaded_models
