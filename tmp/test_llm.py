from llama_index.core.llms import ChatMessage, TextBlock, ImageBlock
from llama_index.llms.openai_like import OpenAILike

llm = OpenAILike(
    model="qwen3-vl-plus",
    is_chat_model=True,
    is_function_calling_model=True 
)

messages = [
    ChatMessage(
        role="user",
        blocks=[
            ImageBlock(path=r"data_charts_gen\01_sales_trend_poisoned.png"),
            TextBlock(text="Describe the image in a few sentences."),
        ],
    )
]

resp = llm.chat(messages)
print(resp.message.content)