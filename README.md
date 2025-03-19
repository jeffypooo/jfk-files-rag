# 🕵️‍♂️ Who Killed JFK? Let's Find Out! 🔍

I hacked this project together in a few hours. It's a simple RAG system that uses Weaviate and OpenAI to answer questions about the recently released JFK assassination files. 

## Credits

This project uses the JFK Files dataset originally compiled by [Amjad Masad](https://github.com/amasad/jfk_files).

## Prerequisites

- Python 3.8+
- Weaviate Cloud account
- OpenAI API key

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/jfk-files-rag.git
   cd jfk-files-rag
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

3. Install the dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Set up your environment variables:
   - Copy `.env.example` to `.env`
   - Fill in your Weaviate and OpenAI credentials:
     ```
     WEAVIATE_URL="your_weaviate_url"
     WEAVIATE_API_KEY="your_weaviate_api_key"
     WEAVIATE_COLLECTION_NAME="JFKFiles"
     OPENAI_API_KEY="your_openai_api_key"
     OPENAI_MODEL="gpt-4.5-preview"  # or any other OpenAI model. 4.5 was very impressive for this.
     ```

## Setting Up External Services

### Weaviate

1. Create a [Weaviate Cloud](https://weaviate.io/pricing) account
2. Create a new cluster, sandbox is fine
3. Get your cluster URL and API key
4. Add them to your `.env` file

### OpenAI

1. Sign up for an [OpenAI API account](https://platform.openai.com/signup)
2. Generate an API key 
3. Add the key to your `.env` file

## Running the Application

1. Make sure you've activated your virtual environment and configured your `.env` file
2. Run the application:
   ```
   python main.py
   ```

3. On first run, the application will:
   - Create a Weaviate collection
   - Process and chunk the JFK files
   - Index the chunks in the vector database

4. After initialization, you can start asking getting your tinfoil hat on with your personal AI conspiracy theorist.

## Configuration

You can adjust the following settings in the `.env` file:

- `DELETE_COLLECTION`: Set to 1 to recreate the Weaviate collection on each run
- `CHUNK_SIZE_WORDS`: Size of text chunks in words (default: 200)
- `OVERLAP_SIZE_WORDS`: Overlap between adjacent chunks (default: 50)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

