"""llava.py.

File for providing the Llava model implementation.
"""
from transformers import LlavaForConditionalGeneration

from src.models.base import ModelBase
from src.models.config import Config


class LlavaModel(ModelBase):
    """Llava model implementation."""

    def __init__(self, config: Config) -> None:
        """Initialization of the llava model.

        Args:
            config (Config): Parsed config
        """
        # initialize the parent class
        super().__init__(config)

    def _load_specific_model(self) -> None:
        """Overridden function to populate self.model."""
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.model_path, **self.config.model
        ) if hasattr(self.config, 'model') else (
            LlavaForConditionalGeneration.from_pretrained(
                self.model_path
            )
        )

    def get_vision_key(self):
        return "vision_tower"

    def get_layer_modality(self, layer_name) -> str:
        """Returns 'vision' or 'text' depending on which part of the model the layer is from"""
        if layer_name.startswith("vision_tower") or layer_name.startswith("multi_modal_projector"):
            return "vision"
        elif layer_name.startswith("language_model"):
            return "text"
        else:
            raise ValueError("Layer '{}' not recognized for LLaVA".format(layer_name))