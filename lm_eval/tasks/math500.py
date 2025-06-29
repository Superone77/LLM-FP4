from lm_eval.base import Task, rf
from lm_eval.metrics import mean


class Math500(Task):
    """Subset of mathematics problems with 500 evaluation questions."""

    VERSION = 0
    DATASET_PATH = "HuggingFaceH4/MATH-500"
    DATASET_NAME = "default"

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def test_docs(self):
        return self.dataset["test"]

    def doc_to_text(self, doc):
        return "Problem: " + doc["problem"] + "\nAnswer:"

    def doc_to_target(self, doc):
        return " " + doc["solution"]

    def construct_requests(self, doc, ctx):
        return rf.greedy_until(ctx, {"until": ["\n"]})

    def process_results(self, doc, results):
        pred = results[0].strip()
        gold = doc["solution"].strip()
        return {"acc": int(pred == gold)}

    def aggregation(self):
        return {"acc": mean}

    def higher_is_better(self):
        return {"acc": True}
