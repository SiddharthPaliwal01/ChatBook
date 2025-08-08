# 📚 ChatBook - Advanced PDF Question Answering System

ChatBook is a production-ready, AI-powered PDF question-answering system that combines multiple state-of-the-art models to provide accurate and contextual answers from your documents.

## 🌟 Features

- **Multi-Model Architecture**: Combines local transformers models with cloud-based LLMs
- **Smart Model Selection**: Automatically chooses the best model based on question complexity
- **Advanced Document Processing**: Intelligent chunking and vector embeddings
- **Beautiful Web Interface**: Modern Streamlit UI with real-time responses
- **Production Ready**: Docker support, logging, error handling, and monitoring
- **Flexible Configuration**: Customizable models, parameters, and settings

## 🏗️ Architecture

```
ChatBook/
├── app/                    # Main application package
│   ├── functions.py        # Core ChatBook class and logic
│   ├── streamlit_app.py    # Web interface
│   └── utils/              # Utility modules
│       ├── exceptions.py   # Custom exceptions
│       └── logging.py      # Logging configuration
├── config.py              # Configuration management
├── data/                  # PDF documents storage
├── logs/                  # Application logs
├── vectorstore/           # Document embeddings storage
└── requirements-prod.txt  # Production dependencies
```

## 🚀 Quick Start

### Option 1: Automated Setup (Recommended)

```bash
# Clone the repository
git clone https://github.com/SiddharthPaliwal01/ChatBook.git
cd ChatBook

# Run the setup script
python setup.py

# Copy and configure environment
cp .env.example .env
# Edit .env and add your OpenRouter API key

# Start the application
./start.sh  # Linux/Mac
# or
start.bat   # Windows
```

### Option 2: Manual Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements-prod.txt

# Set environment variables
export OPENROUTER_API_KEY="your_api_key_here"

# Run the application
streamlit run app/streamlit_app.py
```

### Option 3: Docker Deployment

```bash
# Copy environment file
cp .env.example .env
# Edit .env with your API key

# Build and run with Docker Compose
docker-compose up --build

# Or run with plain Docker
docker build -t chatbook .
docker run -p 8501:8501 --env-file .env chatbook
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file with the following variables:

```bash
# Required
OPENROUTER_API_KEY=your_openrouter_api_key_here

# Optional - Model Configuration
LLM_MODEL=anthropic/claude-3.5-sonnet
QA_MODEL=deepset/roberta-base-squad2
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Optional - Processing Parameters
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
RETRIEVER_K=5
MIN_CONFIDENCE_THRESHOLD=0.3
MAX_CONTEXT_LENGTH=512
RATE_LIMIT_REQUESTS=60
```

## 📖 Usage

### Web Interface

1. **Upload PDF**: Choose a PDF document to analyze
2. **Configure Models**: Select your preferred AI models in the sidebar
3. **Process Document**: Click "Process Document" to create embeddings
4. **Ask Questions**: Type questions or use suggested prompts
5. **View Results**: Get answers with confidence scores and source context

### Programmatic Usage

```python
from config import Config
from app.functions import ChatBook

# Initialize
config = Config()
chatbook = ChatBook(config)

# Load document
chatbook.load_document("path/to/document.pdf")

# Ask questions
result = chatbook.answer_question("What is the main topic?")
print(f"Answer: {result['answer']}")
print(f"Confidence: {result['confidence']}")
```

## 🧪 Testing

Run the test suite to verify everything works:

```bash
# Set your API key
export OPENROUTER_API_KEY="your_key_here"

# Run tests
python test_chatbook.py
```

## 🚀 Getting Started

Ready to use ChatBook? Follow these steps:

1. **Get an OpenRouter API Key**: Sign up at [OpenRouter.ai](https://openrouter.ai)
2. **Run Setup**: `python setup.py`
3. **Configure**: Add your API key to `.env`
4. **Start**: Run `./start.sh` or `start.bat`
5. **Open**: Visit http://localhost:8501

## 🎯 Supported Models

### Cloud LLMs (via OpenRouter)
- **Claude 3.5 Sonnet** (Recommended for complex analysis)
- **GPT-4 Turbo** (Great for general questions)
- **Llama 3.1** (Fast and efficient)

### Local Models
- **RoBERTa-base-squad2** (Default QA model)
- **DistilBERT** (Faster alternative)
- **Sentence Transformers** (For embeddings)

## 📊 Performance Features

- **Parallel Processing**: Multiple chunks processed simultaneously
- **Caching**: LRU cache for repeated questions
- **Rate Limiting**: Respects API limits automatically
- **Smart Fallback**: Falls back to local models if cloud models fail
- **Answer Quality Metrics**: Confidence scores and consistency analysis

## 🛠️ Development

### Project Structure

```
app/
├── functions.py           # Core ChatBook implementation
├── streamlit_app.py      # Web interface
└── utils/
    ├── exceptions.py     # Custom exception classes
    └── logging.py        # Logging configuration

config.py                 # Configuration management
setup.py                 # Production setup script
test_chatbook.py         # Test suite
```

## 🔒 Security & Privacy

- **Local Processing**: Documents processed locally when possible
- **Secure API Calls**: Encrypted communication with cloud providers
- **No Data Storage**: Cloud providers don't store your documents
- **Environment Variables**: Secure API key management

## 📈 Monitoring & Logging

ChatBook includes comprehensive logging:

- **Application Logs**: Stored in `logs/chatbook.log`
- **Error Tracking**: Detailed error messages and stack traces
- **Performance Metrics**: Response times and model usage

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Commit your changes: `git commit -am 'Add feature'`
5. Push to the branch: `git push origin feature-name`
6. Submit a pull request


## 🙏 Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) for document processing
- [Streamlit](https://streamlit.io/) for the web interface
- [OpenRouter](https://openrouter.ai/) for API access to multiple LLMs
- [Hugging Face Transformers](https://huggingface.co/transformers/) for local models

---

Made with ❤️ by [Siddharth Paliwal](https://github.com/SiddharthPaliwal01)
