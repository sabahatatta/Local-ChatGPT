import chainlit as cl
import ollama
import asyncio
import os
import sys
from absl import flags

# Initialize absl flags for gRPC logging
flags.FLAGS(sys.argv[:1])  # Pass only the program name

# Set environment variables for gRPC debugging (optional)
os.environ['GRPC_VERBOSITY'] = 'DEBUG'
os.environ['GRPC_TRACE'] = 'all'

@cl.on_chat_start
async def start_chat():
    # Initialize the interaction history in the user session
    cl.user_session.set(
        "interaction",
        [
            {
                "role": "system",
                "content": "You are a helpful assistant.",
            }
        ],
    )
    
    # Send a welcome message to the user (streamed token-by-token)
    msg = cl.Message(content="")
    start_message = "Hello, I'm your local alternative to ChatGPT running on Llama3.2-Vision. How can I help you today?"
    for token in start_message:
        await msg.stream_token(token)
    await msg.send()

@cl.step(type="tool")
async def tool(input_message, images=None):
    # Retrieve the interaction history from the user session
    interaction = cl.user_session.get("interaction")
    
    # Append the user's message to the interaction history (without image data)
    interaction.append({"role": "user", "content": input_message})
    
    # Prepare image data if provided (read binary data from each image file)
    image_data = None
    if images:
        image_data = []
        for path in images:
            try:
                with open(path, "rb") as f:
                    image_data.append(f.read())
            except Exception as e:
                print(f"Error reading image at {path}: {e}")
    
    # Call the Ollama chat API, passing images separately if available
    # (If the API expects images inside the conversation history, adjust accordingly.)
    response = ollama.chat(model="llama3.2-vision", messages=interaction, images=image_data)
    
    # Append the assistant's response to the interaction history
    interaction.append({"role": "assistant", "content": response["message"]["content"]})
    
    return response

@cl.on_message
async def main(message: cl.Message):
    # Extract images using a stricter check (mime type starts with "image/")
    images = [file for file in message.elements if file.mime.startswith("image/")]
    
    # Pass image file paths to the tool function if images exist
    if images:
        # Pass the file paths so that the tool function can read and process them
        tool_res = await tool(message.content, [i.path for i in images])
    else:
        tool_res = await tool(message.content)
    
    # Stream the assistant's response back to the user token-by-token
    msg = cl.Message(content="")
    for token in tool_res["message"]["content"]:
        await msg.stream_token(token)
    await msg.send()

# Cleanup function to ensure proper shutdown of resources
def cleanup():
    print("Cleaning up resources...")
    # Add any necessary cleanup logic here, such as closing connections

# Register the cleanup function to run at exit
import atexit
atexit.register(cleanup)

# Main wrapper to manage the asynchronous event loop
async def main_wrapper():
    await cl.main()

if __name__ == "__main__":
    asyncio.run(main_wrapper())