import gradio as gr
# The file `chatservice.py` needs to be in the same directory as `app.py`
from chatservice import ProactiveChatService 

# --- 1. Initialize the Chat Service ---
# This is a critical step. We create ONE instance of the service that will be shared
# across all user sessions in this Gradio app. This is how the conversation history
# is maintained, as the `chat_service` object itself persists.
print("Initializing Chat Service for Gradio App...")

# --- THIS IS THE CORRECTED LINE ---
# The `num_passages_to_retrieve` argument is removed because our new, smarter
# Analyst stage decides this dynamically for each query.
chat_service = ProactiveChatService(history_length=8)
print("Chat Service ready.")


# --- 2. Define the Core Chatbot Function ---
# This function is the bridge between Gradio's UI and our backend chat service.
# Gradio's ChatInterface will call this function every time the user sends a message.
def predict(message: str, history: list):
    """
    The main prediction function that powers the chatbot interface.

    Args:
        message (str): The user's input message.
        history (list): The chat history managed by Gradio (we ignore this and use our service's internal history).

    Yields:
        str: A stream of strings that builds the chatbot's response.
    """
    
    # --- The "Thinking" Indicator ---
    # Immediately yield a placeholder message. This appears instantly in the UI
    # while the Analyst and Retriever stages are running in the background.
    yield "⌛ Thinking..."

    # This list will accumulate the full response as it's generated.
    full_answer_list = []
    
    # --- Call the Chat Service ---
    # The `chat_service.chat()` method is a generator that yields events.
    # We loop through these events to handle streaming and metadata.
    try:
        # We start the process which takes a couple of seconds for the non-streaming part.
        for event in chat_service.chat(message):
            
            if event["type"] == "answer_chunk":
                # If we get an answer chunk, append it to our list.
                full_answer_list.append(event["content"])
                # Yield the complete message so far. Gradio will replace the
                # "Thinking..." message with the first chunk, and then append to it.
                yield "".join(full_answer_list)
            
            # --- THIS BLOCK IS NOW UNCOMMENTED TO HANDLE SOURCES ---
            elif event["type"] == "final_data":
                # Once the main stream is done, we get the final metadata.
                sources = event["content"].get("sources", [])
                if sources:
                    # Append the formatted sources to the end of the message.
                    # This adds functionality without changing the look of the chat bubbles.
                    source_str = "\n\n---\n*তথ্যসূত্র* " + ", ".join(sources)
                    full_answer_list.append(source_str)
                    # Yield the final, complete message including sources.
                    yield "".join(full_answer_list)

            elif event["type"] == "error":
                # If the service reports an error, display it.
                yield event["content"]

    except Exception as e:
        # Catch any unexpected errors during the process.
        print(f"An unexpected error occurred in predict function: {e}")
        yield "I'm sorry, an unexpected error occurred. Please try again."


# --- 3. Configure and Launch the Gradio UI ---
# This section is unchanged to preserve the look you like.
demo = gr.ChatInterface(
    fn=predict,  # The function that powers the bot
    title=" AI Agent",
    description="test",
    # examples=[
    #     ["What kind of beef do you have?"],
    #     ["How much is the mutton leg?"],
    #     ["I want to cook a biryani, what do you recommend?"]
    # ],
    cache_examples=False, # Set to False to ensure the bot's memory is fresh for examples
)

if __name__ == "__main__":
    # Launch the Gradio app with an in-browser link and optional sharing.
    demo.launch(share=True,server_name="0.0.0.0",server_port=3032)