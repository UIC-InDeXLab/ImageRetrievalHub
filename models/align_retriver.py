from models.model_interfaces import BaseRetriever
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel
from typing import List
from tqdm import tqdm


class ALIGNRetriever(BaseRetriever):
    def __init__(self, image_dir: str):
        super().__init__(image_dir)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Load processor and model from the provided HF repo
        self.processor = AutoProcessor.from_pretrained("kakaobrain/align-base")
        self.model = AutoModel.from_pretrained("kakaobrain/align-base").to(self.device)
        # Use the same checkpoint for text processing
        self.text_processor = AutoProcessor.from_pretrained("kakaobrain/align-base")
        self.text_model = AutoModel.from_pretrained("kakaobrain/align-base").to(self.device)

    def preprocess(self) -> None:
        image_features = []
        for img_path in tqdm(self.image_paths, desc="Preprocessing ALIGN images"):
            image = Image.open(img_path).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                # Use get_image_features if available, else fallback to CLS token extraction
                if hasattr(self.model, "get_image_features"):
                    features = self.model.get_image_features(**inputs)
                else:
                    outputs = self.model(**inputs)
                    features = outputs.last_hidden_state[:, 0]  # CLS token
            # Normalize features to unit norm
            features = features / features.norm(dim=1, keepdim=True)
            image_features.append(features)
        self.preprocessed_data["image_features"] = torch.cat(image_features, dim=0)

    def retrieve(self, query: str, n: int = 5) -> List[str]:
        self.initialize()
        text_inputs = self.text_processor(text=query, return_tensors="pt").to(self.device)
        with torch.no_grad():
            if hasattr(self.text_model, "get_text_features"):
                text_features = self.text_model.get_text_features(**text_inputs)
            else:
                outputs = self.text_model(**text_inputs)
                text_features = outputs.last_hidden_state[:, 0]
        # Normalize text features
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        image_features = self.preprocessed_data["image_features"]
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        # Compute cosine similarities (dot product on normalized vectors)
        similarities = torch.matmul(text_features, image_features.T).squeeze(0)
        top_indices = similarities.argsort(descending=True)[:n]
        return [str(self.image_paths[idx]) for idx in top_indices]
