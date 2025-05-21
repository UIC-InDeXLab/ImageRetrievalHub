from typing import List, Tuple

from models.model_interfaces import BaseRetriever
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

from tqdm import tqdm


class CLIPRetriever(BaseRetriever):
    def __init__(self, image_dir: str):
        super().__init__(image_dir)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def preprocess(self) -> None:
        image_features = []

        for img_path in tqdm(self.image_paths):
            image = Image.open(img_path).convert('RGB')
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)

            with torch.no_grad():
                image_features_batch = self.model.get_image_features(**inputs)
                # Normalize features
                image_features_batch = image_features_batch / image_features_batch.norm(dim=1, keepdim=True)
                image_features.append(image_features_batch)

        # Concatenate all features
        self.preprocessed_data["image_features"] = torch.cat(image_features)

    def retrieve(self, query: str, n: int = 5) -> List[Tuple[str, float]]:
        self.initialize()

        # Process text query
        inputs = self.processor(text=query, return_tensors="pt", padding=True).to(self.device)

        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
            # Normalize features
            text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # Calculate similarities
        similarities = torch.matmul(text_features, self.preprocessed_data["image_features"].T).squeeze(0)

        # Get top n matches
        top_indices = similarities.argsort(descending=True)[:n]
        scores = similarities.sort(descending=True)[:n]
        return [(str(self.image_paths[idx]), float(score)) for idx, score in zip(top_indices, scores)]
