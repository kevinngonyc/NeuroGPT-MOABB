from train_gpt import make_model, get_config
from sklearn.pipeline import make_pipeline
from moabb.evaluations import WithinSessionEvaluation
from moabb.paradigms import MotorImagery
from moabb.datasets import AlexMI

model = make_model(get_config())
model.from_pretrained("pytorch_model.bin")

pipelines = {"NeuroGPT": make_pipeline(model)}
datasets = [AlexMI()]
paradigm = MotorImagery(n_classes=3)

evaluation = WithinSessionEvaluation(
    paradigm=paradigm, 
    datasets=datasets,
)

results = evaluation.process(pipelines)
