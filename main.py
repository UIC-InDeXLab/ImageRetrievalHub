from enum import Enum
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from models.align_retriver import ALIGNRetriever
from models.blip_retriever import BLIPRetriever
from models.clip_retriever import CLIPRetriever
from models.flava_retriever import FLAVARetriever


class ModelType(str, Enum):
    CLIP = "clip"
    BLIP = "blip"
    FLAVA = "flava"
    ALIGN = "align"


class RetrievalRequest(BaseModel):
    query: str
    n: int = 60
    model: ModelType


app = FastAPI(title="Image Retrieval API")

# Initialize retrievers with image directory
IMAGE_DIR = "/home/mahdi/datasets/COCO/train2017/"

retrievers = {
    ModelType.CLIP: CLIPRetriever(IMAGE_DIR),
    ModelType.BLIP: BLIPRetriever(IMAGE_DIR),
    ModelType.FLAVA: FLAVARetriever(IMAGE_DIR),
    ModelType.ALIGN: ALIGNRetriever(IMAGE_DIR),
}


@app.post("/retrieve/", response_model=List[str])
async def retrieve_images(request: RetrievalRequest):
    """
    Retrieve the top n most relevant images for the given query using the specified model.
    """
    try:
        retriever = retrievers[request.model]
        results = retriever.retrieve(request.query, request.n)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models/", response_model=List[str])
async def list_models():
    """
    List all available retrieval models.
    """
    return [model.value for model in ModelType]


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8020)
