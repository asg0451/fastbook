
from bentoml.frameworks.fastai import FastaiModelArtifact
from bentoml.adapters import FileInput
from fastcore.utils import tuplify, detuplify

import bentoml
import torchvision # not imported by default, to help with pickling
# import datablock_utils

# img conversion
from fastai.vision.core import PILImage

@bentoml.artifacts([FastaiModelArtifact('learner')])
@bentoml.env(infer_pip_packages=True)
class PetClassificationService(bentoml.BentoService):
    @bentoml.api(input=FileInput(), batch=True)
    def predict(self, files):
        # TODO: learner.get_preds
        results = [self.artifacts.learner.predict(PILImage.create(i)) for i in files]
        return results
