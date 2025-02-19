from typing import List
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
                # Assume outputs shape is [1, num_tokens, hidden_size] (e.g. [1, 197, D]).
                # Pool by taking the first token (CLS token).
                features = outputs[:, 0, :]  # Now shape is [1, hidden_size]
            image_features.append(features)

        # Concatenate features along batch dimension; result shape: [num_images, hidden_size]
        self.preprocessed_data["image_features"] = torch.cat(image_features, dim=0)

    def retrieve(self, query: str, n: int = 5) -> List[str]:
        self.initialize()

        # Process the text query
        text_inputs = self.processor(text=query, return_tensors="pt").to(self.device)

        with torch.no_grad():
            text_features = self.model.get_text_features(**text_inputs)
            # Assume text_features shape is [1, num_tokens, hidden_size] (e.g. [1, 6, D]).
            # Pool by taking the first token.
            text_features = text_features[:, 0, :]  # Now shape is [1, hidden_size]

        # Expand text_features so it can be compared to each image feature
        num_images = self.preprocessed_data["image_features"].shape[0]
        text_features_expanded = text_features.expand(num_images, -1)  # Shape: [num_images, hidden_size]

        # Compute cosine similarity between the text vector and each image vector
        similarities = F.cosine_similarity(self.preprocessed_data["image_features"], text_features_expanded, dim=1)

        # Get indices of the top n most similar images
        top_indices = similarities.argsort(descending=True)[:n]
        return [str(self.image_paths[idx]) for idx in top_indices]
