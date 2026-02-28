from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from openai import OpenAI
import os

app = FastAPI()

# Use AI Pipe instead of direct OpenAI
client = OpenAI(
    api_key=os.getenv("AIPIPE_API_KEY"),
    base_url="https://aipipe.org/openai/v1"
)

# -------- Request Model --------
class CommentRequest(BaseModel):
    comment: str = Field(..., min_length=1)

# -------- Response Model --------
class SentimentResponse(BaseModel):
    sentiment: str
    rating: int

# -------- Endpoint --------
@app.post("/comment", response_model=SentimentResponse)
def analyze_comment(request: CommentRequest):
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a sentiment analysis API. Return only valid JSON."
                },
                {
                    "role": "user",
                    "content": request.comment
                }
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "sentiment_schema",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "sentiment": {
                                "type": "string",
                                "enum": ["positive", "negative", "neutral"]
                            },
                            "rating": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 5
                            }
                        },
                        "required": ["sentiment", "rating"],
                        "additionalProperties": False
                    }
                }
            }
        )

        # AI Pipe returns JSON as string → we parse manually
        import json
        content = response.choices[0].message.content
        result = json.loads(content)

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))