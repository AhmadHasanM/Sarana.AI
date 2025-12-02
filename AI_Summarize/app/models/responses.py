from pydantic import BaseModel

class SummaryResponse(BaseModel):
    summary: str
    page_count: int
    processed_images: int