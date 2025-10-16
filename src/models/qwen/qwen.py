"""qwen.py.

File for providing the Qwen model implementation.
"""
from transformers import Qwen2VLForConditionalGeneration

from src.models.base import ModelBase
from src.models.config import Config


class QwenModel(ModelBase):
    """Qwen model implementation."""

    def __init__(self, config: Config) -> None:
        """Initialization of the qwen model.

        Args:
            config (Config): Parsed config
        """
        # initialize the parent class
        super().__init__(config)

    def _load_specific_model(self) -> None:
        """Overridden function to populate self.model."""
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_path, **self.config.model
        ) if hasattr(self.config, 'model') else (
            Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_path
            )
        )

    def get_vision_key(self) -> str:
        return "visual"

    def get_layer_modality(self, layer_name) -> str:
        """Returns 'vision' or 'text' depending on which part of the model the layer is from"""
        if layer_name.startswith("visual"):
            return "vision"
        elif layer_name.startswith("model"):
            return "text"
        else:
            raise ValueError("Layer '{}' not recognized for Qwen".format(layer_name))
