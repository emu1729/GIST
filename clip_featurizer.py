import functools
import logging
from threading import Lock
import time
import PIL.Image
import torch
import clip

logger = logging.getLogger(__name__)


def log_duration(function):
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        start = time.time()
        results = function(*args, **kwargs)
        end = time.time()
        logger.debug(f"{function.__name__} completed. processing_time={round(end-start, 3)}s")
        return results
    return wrapper


class ClipFeaturizer:
    def __init__(
        self,
        ic_name: str = 'ViT-B/32',
        device = torch.device('cpu'),
        use_logit_scale: bool = False,
        weights_path: str = None,
        ):
        """Constructor

        Args:
            ic_name (str, optional): Name of the model. Defaults to 'Florence-v1.1-davit-d5'.
            ic_weights_filepath (str, optional): Weight filepath. Defaults to 'florence_v1.1_davit_d5.pth'.
            device (torch.device, optional): Torch device. Defaults to torch.device('cpu').
            use_logit_scale (bool, optional): True to use logit scale for computing image features. Defaults to False.
        """
        self.use_logit_scale = use_logit_scale
        self._ic_model, self._ic_preprocess = clip.load(ic_name, device=device)
        self._logit_scale = self._ic_model.logit_scale.exp()
        if weights_path:
            state_dict = torch.load(weights_path)
            self._ic_model.load_state_dict(state_dict)
        self._ic_model.to(device)
        self.device = device
        self._lock = Lock()

    @torch.no_grad()
    @log_duration
    def compute_image_feature(self, image: PIL.Image) -> torch.Tensor:
        """Compute a feature vector for an input image."""
        image = image.convert('RGB')
        inputs = self._ic_preprocess(image).unsqueeze(0).to(self.device)
        with self._lock:
            if self.use_logit_scale:
                return self._ic_model.encode_image(inputs) * self._logit_scale
            else:
                return self._ic_model.encode_image(inputs)

    @torch.no_grad()
    @log_duration
    def compute_text_feature(self, text: str) -> torch.Tensor:
        """Compute a feature vector for input text."""
        with self._lock:
            tokens = clip.tokenize(text, context_length=77).to(self.device)
            return self._ic_model.encode_text(tokens)