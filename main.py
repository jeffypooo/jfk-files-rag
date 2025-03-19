import os
import re
import time
import asyncio
from queue import Queue
import dotenv
import weaviate
import weaviate.classes as wvc
from weaviate.collections import Collection
import logging
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
    # verify connection
    try:
        client.models.list()
    except Exception as e:
        raise e
    return client

def initialize_weaviate_client() -> weaviate.Client:
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=config.WEAVIATE_URL,
        auth_credentials=wvc.init.Auth.api_key(config.WEAVIATE_API_KEY),
        headers={
            "X-OpenAI-Api-Key": config.OPENAI_API_KEY,
        }
    )
    if config.DEBUG:
        try:
            metadata = client.get_meta()
        except Exception as e:
            raise e
    return client

def create_collection(client: weaviate.Client):
    with console.status("Creating collection...", spinner="bouncingBall") as status:
        if config.ALWAYS_RECREATE_COLLECTION:
            status.update("Deleting collection...")
            client.collections.delete(config.WEAVIATE_COLLECTION_NAME)
            # ensure the collection is deleted
            time.sleep(1)

        status.update("Checking if collection exists...")
        collection = client.collections.exists(config.WEAVIATE_COLLECTION_NAME)
        if collection:
            return

        status.update("Creating collection...")
        client.collections.create(
            name=config.WEAVIATE_COLLECTION_NAME,
            properties=[
                wvc.config.Property(
                    name="chunk",
                    data_type=wvc.config.DataType.TEXT,
                ),
                wvc.config.Property(
                    name="file_name",
                    data_type=wvc.config.DataType.TEXT,
                ),
                wvc.config.Property(
                    name="chunk_index",
                    data_type=wvc.config.DataType.INT,
                ),
            ],
            vectorizer_config=wvc.config.Configure.Vectorizer.text2vec_openai(),
    )

def process_files(files: list[str], client: weaviate.Client, progress: rich.progress.Progress, task_id: rich.progress.TaskID):
    """
    Process a file list and insert the chunks of each into the collection.
    """
    collection: Collection = client.collections.get(config.WEAVIATE_COLLECTION_NAME)
    for i, file in enumerate(files):
        with open(os.path.join("jfk_text", file), "r") as f:
            text = f.read()
        # remove instances of multiple whitespace characters
        text = re.sub(r'\s+', ' ', text)
        # split text into words
        words = re.split(r'\s', text)
        # remove empty strings
        words = [word for word in words if len(word) > 0]
        # split words into strings of words of size config.CHUNK_SIZE_WORDS + config.OVERLAP_SIZE_WORDS
        # this uses a sliding window approach to create chunks
        chunks = []
        for i in range(0, len(words), config.CHUNK_SIZE_WORDS):
            chunk = " ".join(words[max(0, i - config.OVERLAP_SIZE_WORDS):i + config.CHUNK_SIZE_WORDS])
            chunks.append(chunk)
        to_insert = list()
        for i, chunk in enumerate(chunks):
            data_properties = {
                "chunk": chunk,
                "file_name": file,
                "chunk_index": i,
            }
            data_obj = wvc.data.DataObject(properties=data_properties)
            to_insert.append(data_obj)
        collection.data.insert_many(to_insert)
        progress.update(task_id, advance=1)

def add_data_to_collection(client: weaviate.Client):
    if not config.ALWAYS_RECREATE_COLLECTION:
        return
    # Text files are in the jfk_text directory
    files = os.listdir("jfk_text")
    # split files in chunks based on the number of CPUs
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

def search_jfk_files(query: str, client: weaviate.Client):
    collection: Collection = client.collections.get(config.WEAVIATE_COLLECTION_NAME)
    return collection.query.near_text(
        query=query,
        limit=10,
        return_metadata=wvc.query.MetadataQuery(distance=True)
    )

def main():
    with console.status("Initializing clients...", spinner="bouncingBall"):
        openai_client = initialize_openai_client()
        weaviate_client = initialize_weaviate_client()

    try:
        # Create collection if needed.
        create_collection(weaviate_client)
        add_data_to_collection(weaviate_client)
        console.print("Welcome to the JFK Files! Ready to find out the truth?", style="bold white")

        # tools for the search agent
        tools = [
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
                },
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_name": {"type": "string", "description": "The name of the file to read"},
                    },
                    "required": ["file_name"],
                },
            }
        ]
        # system prompt for the search agent
        system_prompt = """
        You are a search agent that can search for information about the JFK assassination.

        The user will ask you questions about the JFK assassination, and you will use the tools provided to answer the questions.

        When using the search_jfk_files tool, take care to craft a query that will return the most relevant results, given that you are
        searching via a vector database. There are approximately 1000 files in the database, containing recently declassified documents
        related to the JFK assassination, the investigation, and the legacy of the Kennedy family.
        """


        messages = [
            {"role": "system", "content": system_prompt},
        ]
        while True:
            user_query = console.input("Enter a query, or 'q' to quit: ")
            if user_query == "q":
                break
            messages.append({"role": "user", "content": user_query})
            # invoke the search agent
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                tools=tools,
                tool_choice="auto",
                stream=True,
            )
            accumulated_content = ""
            tool_calls = []
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    accumulated_content += content
                    console.print(content, end="", style="bold white")
                if chunk.choices[0].delta.tool_calls:
                    tool_call_delta = chunk.choices[0].delta.tool_calls[0]
                    if tool_call_delta.index == len(tool_calls):
                        # new tool call
                        tool_calls.append({"id": tool_call_delta.id, "function": {"name": tool_call_delta.function.name, "arguments": ""}})
                    else:
                        # append arguments incrementally
                        tool_calls[tool_call_delta.index]["function"]["arguments"] += tool_call_delta.function.arguments

            if len(accumulated_content) > 0:
                console.print("\n")
            if len(tool_calls) > 0:
                for tool_call in tool_calls:
                    if tool_call["function"]["name"] == "search_jfk_files":
                        with console.status("Searching JFK Files...", spinner="bouncingBall") as status:
                            function_call_ouput = ""
                            query = json.loads(tool_call["function"]["arguments"])["query"]
                            # search weaviate for the query
                            results = search_jfk_files(query, weaviate_client)
                            sorted_results = sorted(results.objects, key=lambda x: x.metadata.distance)
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
                                function_call_ouput += object_output
                            messages.append({"role": "tool", "content": function_call_ouput, "tool_call_id": tool_call["id"]})

    except Exception as e:
        print(f"something went wrong: {e}")
        raise e
    finally:
        weaviate_client.close()

if __name__ == "__main__":
    main()