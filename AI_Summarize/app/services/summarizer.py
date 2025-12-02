from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from typing import List, Dict
import base64
from io import BytesIO
from ..config.settings import get_settings
from .prompt_builder import get_summary_prompt

settings = get_settings()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=settings.GOOGLE_API_KEY,
    temperature=0.1,
    convert_system_prompt_to_human=True
)

async def summarize_pdf(contents: List[Dict]) -> str:
    message_contents = []

    for item in contents:
        if item["type"] == "text":
            message_contents.append(item["content"])
        elif item["type"] == "image":
            buffered = BytesIO()
            item["image"].save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            message_contents.append({
                "type": "image_url",
                "image_url": f"data:image/png;base64,{img_str}"
            })

    if not message_contents:
        return "Tidak ada konten yang dapat diproses dari PDF."

    #
    human_message = HumanMessage(content=message_contents)
    prompt = get_summary_prompt()
    chain = prompt | llm
    
    response = await chain.ainvoke({"content": human_message})
    return response.content
