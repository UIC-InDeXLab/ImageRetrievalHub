from typing import List, Tuple

import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import BlipProcessor, BlipForConditionalGeneration

from models.model_interfaces import BaseRetriever


class BLIPRetriever(BaseRetriever):
    def __init__(self, image_dir: str):
        super().__init__(image_dir)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Initialize BLIP for image captioning
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        ).to(self.device)
        # Initialize sentence transformer for text embeddings
        self.sentence_encoder = SentenceTransformer('all-MiniLM-L6-v2').to(self.device)

    def preprocess(self) -> None:
        image_captions = {}
        caption_embeddings = []

        for img_path in tqdm(self.image_paths):
            # Generate caption for the image
            image = Image.open(img_path)
            inputs = self.processor(image, return_tensors="pt").to(self.device)

            with torch.no_grad():
                generated_ids = self.model.generate(**inputs)
                caption = self.processor.decode(generated_ids[0], skip_special_tokens=True)

            # Store caption
            image_captions[str(img_path)] = caption
            print(f"{str(img_path)}: {caption}")

            # Generate embedding for the caption
            with torch.no_grad():
                caption_embedding = self.sentence_encoder.encode(caption, convert_to_tensor=True)
                caption_embeddings.append(caption_embedding)

        # Store both captions and their embeddings
        self.preprocessed_data["captions"] = image_captions
        self.preprocessed_data["caption_embeddings"] = torch.stack(caption_embeddings)

    def retrieve(self, query: str, n: int = 5) -> List[Tuple[str, float]]:
        self.initialize()

        # Generate embedding for the query
        with torch.no_grad():
            query_embedding = self.sentence_encoder.encode(query, convert_to_tensor=True)

        # Calculate cosine similarities between query and all captions
        similarities = torch.cosine_similarity(
            query_embedding.unsqueeze(0),
            self.preprocessed_data["caption_embeddings"]
        )

        # Get top n matches
        top_indices = similarities.argsort(descending=True)[:n]
        scores = similarities.sort(descending=True)[:n]
        return [(str(self.image_paths[idx]), float(score)) for idx, score in zip(top_indices, scores)]
