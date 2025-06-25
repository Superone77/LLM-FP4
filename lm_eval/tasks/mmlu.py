from .hendrycks_test import GeneralHendrycksTest

class MMLU(GeneralHendrycksTest):
    """Massive Multitask Language Understanding (all subjects)."""

    def __init__(self):
        # "all" is the configuration on the HuggingFace dataset that
        # contains the full evaluation set across subjects.
        super().__init__("all")
