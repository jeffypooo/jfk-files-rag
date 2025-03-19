import os
import re
import time
from queue import Queue
import weaviate
import weaviate.classes as wvc
from weaviate.collections import Collection
import json
import threading
import multiprocessing
import numpy as np
import openai
from config import Config
import rich.progress
from rich.console import Console
import textwrap

config = Config()
console = Console()

def initialize_openai_client() -> openai.OpenAI:
    client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
    try:
        client.models.list()
    except Exception as e:
        raise e
    return client

def initialize_weaviate_client() -> weaviate.Client:
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=config.WEAVIATE_URL,
        auth_credentials=wvc.init.Auth.api_key(config.WEAVIATE_API_KEY),
        headers={"X-OpenAI-Api-Key": config.OPENAI_API_KEY}
    )
    return client

def create_collection(client: weaviate.Client) -> bool:
    """
    Create a collection in Weaviate if it doesn't exist.
    Returns True if the collection was created, False if it already exists.
    """
    with console.status("Creating collection...", spinner="bouncingBall") as status:
        if config.DELETE_COLLECTION:
            status.update("Deleting collection...")
            client.collections.delete(config.WEAVIATE_COLLECTION_NAME)
            time.sleep(1)
        status.update("Checking if collection exists...")
        collection = client.collections.exists(config.WEAVIATE_COLLECTION_NAME)
        if collection:
            return False
        status.update("Creating collection...")
        client.collections.create(
            name=config.WEAVIATE_COLLECTION_NAME,
            properties=[
                wvc.config.Property(name="chunk", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="file_name", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="chunk_index", data_type=wvc.config.DataType.INT),
            ],
            vectorizer_config=wvc.config.Configure.Vectorizer.text2vec_openai(),
        )
        return True

def process_files(files: list[str], client: weaviate.Client, progress: rich.progress.Progress, task_id: rich.progress.TaskID):
    collection: Collection = client.collections.get(config.WEAVIATE_COLLECTION_NAME)
    for i, file in enumerate(files):
        with open(os.path.join("jfk_text", file), "r") as f:
            text = f.read()
        text = re.sub(r'\s+', ' ', text)
        words = re.split(r'\s', text)
        words = [word for word in words if len(word) > 0]
        chunks = []
        for i in range(0, len(words), config.CHUNK_SIZE_WORDS):
            chunk = " ".join(words[max(0, i - config.OVERLAP_SIZE_WORDS):i + config.CHUNK_SIZE_WORDS])
            chunks.append(chunk)
        to_insert = []
        for i, chunk in enumerate(chunks):
            data_properties = {"chunk": chunk, "file_name": file, "chunk_index": i}
            data_obj = wvc.data.DataObject(properties=data_properties)
            to_insert.append(data_obj)
        collection.data.insert_many(to_insert)
        progress.update(task_id, advance=1)

def add_data_to_collection(client: weaviate.Client):
    files = os.listdir("jfk_text")
    cpu_count = multiprocessing.cpu_count()
    chunks = np.array_split(files, cpu_count)
    thread_count = len(chunks)
    threads = []
    tasks = Queue(thread_count)
    console.print(f"Ingesting {len(files)} files into Weaviate...", style="bold white")
    with rich.progress.Progress(
        rich.progress.TextColumn("[progress.description]{task.description}"),
        rich.progress.BarColumn(),
        rich.progress.TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        rich.progress.TextColumn("[progress.completed]{task.completed} of {task.total}"),
        rich.progress.TimeRemainingColumn(),
        console=console,
    ) as progress:
        for chunk in chunks:
            task_id = progress.add_task("Processing files", total=len(chunk))
            tasks.put(task_id)
            thread = threading.Thread(target=process_files, args=(chunk, client, progress, task_id))
            threads.append(thread)
            thread.start()
        for thread in threads:
            thread.join()

class ChatHandler:
    def __init__(self, openai_client, weaviate_client):
        self.openai_client = openai_client
        self.weaviate_client = weaviate_client
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_jfk_files",
                    "description": "Search for semantically similar chunks of text in the JFK Files, using a vector search.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "The natural language query to search for"},
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file and return the content",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_name": {"type": "string", "description": "The name of the file to read"},
                        },
                        "required": ["file_name"],
                    },
                },
            }
        ]
        self.system_prompt = """
        You are a search agent that can search for information about the JFK assassination.

        The user will ask you questions about the JFK assassination, and you will use the tools provided to answer the questions.

        When using the search_jfk_files tool, take care to craft a query that will return the most relevant results, given that you are
        searching via a vector database. There are approximately 1000 files in the database, containing recently declassified documents
        related to the JFK assassination, the investigation, and the legacy of the Kennedy family.

        Include quotes from the text in your response, if applicable, including the file name.

        You maintain a skeptical attitude towards the official narrative, and are curious about the truth.
        """
        self.messages = [{"role": "system", "content": self.system_prompt}]

    def start_chat(self):
        console.print("Welcome to the JFK Files! Ready to find out the truth?", style="bold white")
        while True:
            user_query = console.input("Enter a query, or 'q' to quit: ")
            if user_query == "q":
                break
            self.messages.append({"role": "user", "content": user_query})
            self.process_llm_response()

    def process_llm_response(self):
        response = self.openai_client.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=self.messages,
            tools=self.tools,
            stream=True,
        )

        # Initialize assistant_message without tool_calls
        assistant_message = {"role": "assistant", "content": ""}
        tool_calls = []  # Track tool calls separately

        # Process streaming response
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                content_delta = chunk.choices[0].delta.content
                assistant_message["content"] += content_delta
                console.print(content_delta, end="", style="bold white")
            if chunk.choices[0].delta.tool_calls:
                tool_call_delta = chunk.choices[0].delta.tool_calls[0]
                if tool_call_delta.index == len(tool_calls):
                    tool_calls.append({
                        "id": tool_call_delta.id,
                        "type": "function",
                        "function": {"name": tool_call_delta.function.name, "arguments": ""}
                    })
                else:
                    tool_calls[tool_call_delta.index]["function"]["arguments"] += tool_call_delta.function.arguments

        # Only include tool_calls if there are any
        if tool_calls:
            assistant_message["tool_calls"] = tool_calls

        # Append the message to the conversation
        self.messages.append(assistant_message)

        # Print a newline if there was content, for readability
        if assistant_message["content"]:
            console.print("\n")

        # Handle tool calls if present
        if "tool_calls" in assistant_message:
            for tool_call in assistant_message["tool_calls"]:
                with console.status(f"Executing tool: {tool_call['function']['name']}...", spinner="bouncingBall"):
                    tool_output = self.execute_tool(tool_call)
                self.messages.append({
                    "role": "tool",
                    "content": tool_output,
                    "tool_call_id": tool_call["id"]
                })
            # Recurse to continue the conversation
            self.process_llm_response()

    def execute_tool(self, tool_call):
        function_name = tool_call["function"]["name"]
        if function_name == "search_jfk_files":
            return self.search_jfk_files(tool_call)
        elif function_name == "read_file":
            return self.read_file(tool_call)
        else:
            return f"Unknown tool: {function_name}"

    def search_jfk_files(self, tool_call):
        try:
            arguments = json.loads(tool_call["function"]["arguments"])
            query = arguments["query"]
            collection = self.weaviate_client.collections.get(config.WEAVIATE_COLLECTION_NAME)
            results = collection.query.near_text(
                query=query,
                limit=10,
                return_metadata=wvc.query.MetadataQuery(distance=True)
            )
            sorted_results = sorted(results.objects, key=lambda x: x.metadata.distance)
            function_call_output = ""
            for i, o in enumerate(sorted_results):
                object_output = textwrap.dedent(f"""
                Result {i+1}:
                --------------------------------------
                File Name: {o.properties["file_name"]}
                Chunk Index: {o.properties["chunk_index"]}
                Relevance Score: {o.metadata.distance}
                Text: 
                {o.properties["chunk"]}
                --------------------------------------
                """).strip()
                function_call_output += object_output + "\n"
            return function_call_output
        except Exception as e:
            return f"Error in search_jfk_files: {str(e)}"

    def read_file(self, tool_call):
        try:
            arguments = json.loads(tool_call["function"]["arguments"])
            file_name = arguments["file_name"]
            file_path = os.path.join("jfk_text", file_name)
            if not os.path.exists(file_path):
                return f"File {file_name} not found."
            with open(file_path, "r") as f:
                content = f.read()
            return content
        except Exception as e:
            return f"Error in read_file: {str(e)}"

def main():
    with console.status("Initializing clients...", spinner="bouncingBall"):
        openai_client = initialize_openai_client()
        weaviate_client = initialize_weaviate_client()

    try:
        new_collection = create_collection(weaviate_client)
        if new_collection:
            add_data_to_collection(weaviate_client)
        chat_handler = ChatHandler(openai_client, weaviate_client)
        chat_handler.start_chat()
    except Exception as e:
        print(f"something went wrong: {e}")
        raise e
    finally:
        weaviate_client.close()

if __name__ == "__main__":
    main()