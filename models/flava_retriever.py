from typing import List, Tuple
from models.model_interfaces import BaseRetriever
import torch
from PIL import Image
from transformers import FlavaProcessor, FlavaModel
from tqdm import tqdm
import torch.nn.functional as F


class FLAVARetriever(BaseRetriever):
    def __init__(self, image_dir: str):
        super().__init__(image_dir)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = FlavaProcessor.from_pretrained("facebook/flava-full")
        self.model = FlavaModel.from_pretrained("facebook/flava-full").to(self.device)
        self.preprocessed_data = {}  # Ensure this attribute exists

    def preprocess(self) -> None:
        image_features = []

        for img_path in tqdm(self.image_paths):
            # Convert image to RGB to ensure correct channel format
            image = Image.open(img_path).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model.get_image_features(**inputs)
                # Pool by taking the first token (CLS token).
                features = outputs[:, 0, :]  # shape: [1, hidden_size]

            # Detach and move features to CPU to free GPU memory
            features = features.detach().cpu()
            image_features.append(features)

            # Optionally, clear cache every N images:
            # if some_condition:
            #     torch.cuda.empty_cache()

        # Concatenate features along the batch dimension; result shape: [num_images, hidden_size]
        self.preprocessed_data["image_features"] = torch.cat(image_features, dim=0)

    def retrieve(self, query: str, n: int = 5) -> List[Tuple[str, float]]:
        self.initialize()

        # Process the text query
        text_inputs = self.processor(text=query, return_tensors="pt").to(self.device)
        with torch.no_grad():
            text_features = self.model.get_text_features(**text_inputs)
            # Pool by taking the first token (CLS token)
            text_features = text_features[:, 0, :]  # Shape: [1, hidden_size]

        num_images = self.preprocessed_data["image_features"].shape[0]
        text_features_expanded = text_features.expand(num_images, -1)  # Shape: [num_images, hidden_size]

        # Make sure image features are on the same device as text features
        image_features = self.preprocessed_data["image_features"].to(self.device)
        similarities = F.cosine_similarity(image_features, text_features_expanded, dim=1)

        # Get indices of the top n most similar images
        top_indices = similarities.argsort(descending=True)[:n]
        scores = similarities.sort(descending=True)[:n]
        return [(str(self.image_paths[idx]), float(score)) for idx, score in zip(top_indices, scores)]
