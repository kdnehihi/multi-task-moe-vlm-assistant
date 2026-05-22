"""Baseline vision-language model interfaces and wrappers."""

from abc import ABC, abstractmethod
from pathlib import Path


DEFAULT_BLIP_MODEL_NAME = "Salesforce/blip-vqa-base"


class BaselineVLM(ABC):
    """Interface for baseline vision-language QA models."""

    @abstractmethod
    def predict(self, image_path: str, question: str) -> str:
        """Return an answer for one image-question pair."""
        raise NotImplementedError


class DummyBaselineVLM(BaselineVLM):
    """Deterministic placeholder model for testing evaluation pipelines."""

    def __init__(self, default_answer: str = "") -> None:
        self.default_answer = default_answer

    def predict(self, image_path: str, question: str) -> str:
        """Return a fixed answer without reading the image or question."""
        return self.default_answer


class BlipVQABaselineVLM(BaselineVLM):
    """BLIP VQA baseline wrapper using Hugging Face Transformers.

    The model and processor are loaded lazily on the first prediction so that
    tests and lightweight CLI flows can import this module without downloading
    or initializing model weights.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_BLIP_MODEL_NAME,
        device: str | None = None,
        max_new_tokens: int = 20,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.processor = None
        self.model = None

    def predict(self, image_path: str, question: str) -> str:
        """Generate an answer for one image-question pair."""
        self._ensure_loaded()

        import torch
        from PIL import Image

        image = Image.open(Path(image_path)).convert("RGB")
        inputs = self.processor(image, question, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
            )
        answer = self.processor.decode(output_ids[0], skip_special_tokens=True)

        return answer.strip()

    def _ensure_loaded(self) -> None:
        """Load BLIP processor and model if they are not already loaded."""
        if self.processor is not None and self.model is not None:
            return

        import torch
        from transformers import BlipForQuestionAnswering, BlipProcessor

        self.device = self.device or self._select_device(torch)
        self.processor = BlipProcessor.from_pretrained(self.model_name)
        self.model = BlipForQuestionAnswering.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

    def _select_device(self, torch_module) -> str:
        """Select an available inference device."""
        if torch_module.cuda.is_available():
            return "cuda"
        if (
            hasattr(torch_module.backends, "mps")
            and torch_module.backends.mps.is_available()
        ):
            return "mps"
        return "cpu"


def create_baseline_model(
    model_name: str = "dummy",
    model_id: str | None = None,
    device: str | None = None,
) -> BaselineVLM:
    """Create a baseline model wrapper by name.

    TODO:
    - Add Qwen2-VL or LLaVA wrappers if hardware allows.
    - Add config-driven model loading.
    """
    if model_name == "dummy":
        return DummyBaselineVLM()
    if model_name == "blip":
        return BlipVQABaselineVLM(
            model_name=model_id or DEFAULT_BLIP_MODEL_NAME,
            device=device,
        )

    raise ValueError(f"Unsupported baseline model: {model_name}")
