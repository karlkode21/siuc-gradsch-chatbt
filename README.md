# SIUC Graduate School Chatbot

This chatbot answers questions about the SIUC Graduate School based on the content from the Graduate School Catalog.

## Setup Instructions

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file in the project root and add your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

3. Make sure the `SIUC_GS_Catalog.pdf` file is in the project root directory.

## Running the Chatbot

To start the chatbot, run:
```bash
streamlit run app.py
```

The chatbot will open in your default web browser. You can then ask questions about the Graduate School Catalog, and the chatbot will provide relevant answers based on the catalog content.

## Features

- Interactive web interface
- Natural language question answering
- Persistent chat history during the session
- Fast and accurate responses based on the catalog content

## Note

The first time you run the chatbot, it will take a few moments to process the catalog content. This is a one-time process that happens when you first start the application. 