from train_gpt import make_model, get_config
from sklearn.pipeline import make_pipeline
from moabb.evaluations import WithinSessionEvaluation
from moabb.paradigms import LeftRightImagery
from moabb.datasets import BNCI2014_004
from braindecode import EEGClassifier
from braindecode.preprocessing import preprocess, Preprocessor

pytorch_model = make_model(get_config())
pytorch_model.from_pretrained("pytorch_model.bin")
model = EEGClassifier(pytorch_model)

pipelines = {"NeuroGPT": make_pipeline(model)}
datasets = [BNCI2014_004()]
paradigm = LeftRightImagery()

evaluation = WithinSessionEvaluation(
    paradigm=paradigm, 
    datasets=datasets,
)

results = evaluation.process(pipelines)
