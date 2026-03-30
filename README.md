# 🎓 College AI Assistant - Production-Ready Chatbot

A sophisticated, production-ready Python chatbot system for college students and staff, featuring intent classification, transformer-based emotion detection, LLM integration with Groq API, and persistent logging.

## 📋 Features

### Core Features
- **Intent Classification**: TF-IDF + Logistic Regression for accurate intent detection
- **Emotion Detection**: Transformer-based sentiment analysis (DistilBERT)
- **LLM Integration**: Primary Groq API with Ollama local fallback
- **Context-Aware Responses**: Multi-turn conversation support with context management
- **Time Awareness**: Dynamic responses based on time of day
- **Knowledge Base**: Structured JSON database with college information
- **SQLite Logging**: Persistent interaction logging and analytics
- **Modern UI**: Streamlit-based chat interface with debug panel

### Advanced Features
- ✅ Structured prompt engineering with context injection
- ✅ Confidence-based clarification requests
- ✅ Emotion-aware response toning
- ✅ Out-of-scope query detection
- ✅ Automatic API fallback mechanism
- ✅ Conversation analytics and statistics
- ✅ Response caching and optimization
- ✅ Email-ready log export

## 📁 Project Structure

```
AAI_chatbot/
├── app.py                    # Streamlit UI entry point
├── main.py                   # CLI runner (optional)
├── intent_model.py           # Intent classification (scikit-learn)
├── llm_handler.py            # Groq API + Ollama fallback
├── context_manager.py        # Conversation state management
├── emotion_detector.py       # Transformer-based emotion detection
├── database.py               # SQLite logging system
├── utils.py                  # Time awareness & helpers
├── requirements.txt          # Python dependencies
├── .env                      # Environment variables (create this)
├── chatbot.db               # SQLite database (auto-created)
├── intent_model.pkl         # Trained model (auto-created)
├── tfidf_vectorizer.pkl     # TF-IDF vectorizer (auto-created)
└── data/
    ├── intents.json         # Intent training data
    └── college_data.json    # College knowledge base
```

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.8+
- pip (Python package manager)
- (Optional) Ollama for local LLM fallback: https://ollama.ai

### 2. Installation

```bash
# Clone/download the project
cd AAI_chatbot

# Create a virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download transformer model (first run only)
python -c "from transformers import pipeline; pipeline('sentiment-analysis')"
```

### 3. Configuration

Create a `.env` file in the project root:

```env
# Groq API Key (Get from https://console.groq.com)
GROQ_API_KEY=your_groq_api_key_here

# Optional: Ollama configuration
OLLAMA_BASE_URL=http://localhost:11434/api/generate
OLLAMA_MODEL=mistral
```

### 4. Run the Chatbot

**Option A: Streamlit UI (Recommended)**
```bash
streamlit run app.py
```
Then open your browser to `http://localhost:8501`

**Option B: CLI Mode**
```bash
python main.py
```

**Option C: Train Model Only**
```bash
python main.py --train
```

**Option D: View Logs**
```bash
python main.py --logs 50
```

## 🔧 Configuration Guide

### Groq API Setup

1. Go to https://console.groq.com
2. Sign up/login with your account
3. Create an API key
4. Add to `.env` file:
   ```
   GROQ_API_KEY=your_key_here
   ```

### Ollama Fallback (Optional)

For local LLM fallback without Groq API:

```bash
# Install Ollama from https://ollama.ai
# Start Ollama service:
ollama serve

# (In another terminal) Download Mistral model:
ollama pull mistral
```

The chatbot will automatically fall back to Ollama if Groq API fails.

### Customize College Data

Edit `data/college_data.json` with your institution's information:
- Fees structure
- Exam schedules
- Faculty contact information
- Campus facilities
- Holiday calendars
- Placement statistics

### Add New Intents

Edit `data/intents.json` to add new intent categories:

```json
{
  "tag": "new_intent",
  "patterns": [
    "questions related to new intent",
    "more variations here"
  ],
  "responses": ["response templates"]
}
```

Then retrain:
```bash
python main.py --train
```

## 📊 UI Overview

### Main Chat Interface
- Streamlit-based chat similar to ChatGPT
- Real-time message display with typing indicators
- Emoji-enhanced messages for better UX

### Sidebar Features
- **Debug Panel**: Show/hide intent, emotion, confidence, and LLM source
- **Statistics**: Total interactions, average confidence, response times
- **Controls**: Clear history, view logs, export data
- **Model Status**: See trained intent counts and API availability

### Debug Information (when enabled)
- Intent classification details
- Confidence scores
- Detected emotion
- LLM source (Groq/Ollama)
- Response generation time

## 📚 Module Documentation

### `intent_model.py`
- **IntentClassifier**: TF-IDF + Logistic Regression
- Methods: `train()`, `predict()`, `batch_predict()`, `save_model()`, `load_model()`

### `emotion_detector.py`
- **EmotionDetector**: Transformer-based sentiment analysis
- Emotions: happy, stressed, confused, angry, sad, neutral
- Methods: `detect_emotion()`, `batch_detect_emotions()`, `get_emotion_aware_response_tone()`

### `llm_handler.py`
- **LLMHandler**: Groq API + Ollama fallback
- Structured prompt engineering with context injection
- Methods: `generate_response()`, `validate_response()`, `get_stats()`

### `context_manager.py`
- **ConversationContext**: Multi-turn conversation state
- Maintains last 5 conversation turns
- Methods: `add_turn()`, `get_history()`, `get_context_summary()`, `should_ask_clarification()`

### `database.py`
- **ChatbotDatabase**: SQLite logging and analytics
- Methods: `log_interaction()`, `get_logs()`, `get_analytics_summary()`, `export_logs_csv()`

### `utils.py`
- Time awareness: `get_time_of_day()`, `get_time_greeting()`
- Helpers: `normalize_text()`, `truncate_text()`, `calculate_confidence_percentage()`

## 🎯 Usage Examples

### Example 1: Ask About Fees
```
User: How much does engineering cost?
Intent: fees (95%)
Emotion: neutral
Response: [LLM generates accurate fee information]
```

### Example 2: Low Confidence Query
```
User: What's your favorite color?
Intent: unknown (32%)
Response: I'm not entirely sure. Could you provide more details about your question?
```

### Example 3: Out-of-Scope Query
```
User: Tell me about quantum physics
Intent: general (28%)
Response: This is beyond my scope as a college assistant.
```

## 📈 Analytics & Logging

View real-time statistics:
- Total interactions logged
- Average intent confidence
- Emotion distribution
- Top requested topics
- LLM source distribution (Groq vs Ollama)
- Average response times

Export logs:
```python
from database import ChatbotDatabase
db = ChatbotDatabase()
db.export_logs_csv("chatbot_logs.csv")
```

## 🔐 Security Considerations

- ✅ API key stored in `.env` (never commit to git)
- ✅ SQLite database for local logging
- ✅ No user data sent to external services except LLM API
- ✅ Structured input validation
- ✅ Domain restriction for responses

## ⚙️ Performance Optimization

1. **Model Caching**: Intent and emotion models cached in Streamlit
2. **Async API Calls**: Optional async support for concurrent requests
3. **Database Indexing**: Timestamps indexed for fast queries
4. **Prompt Truncation**: Context limited to prevent token overrun
5. **Batch Processing**: Support for multiple simultaneous predictions

## 🐛 Troubleshooting

### Issue: Groq API errors
```
Solution: Check API key in .env, ensure internet connection, verify API quota
```

### Issue: Ollama not connecting
```
Solution: Ensure Ollama is running with: ollama serve
```

### Issue: Transformer model download timeout
```
Solution: Pre-download model: python -c "from transformers import pipeline; pipeline('sentiment-analysis')"
```

### Issue: SQLite database locked
```
Solution: Close all other database connections, restart Streamlit app
```

### Issue: Import errors
```
Solution: Verify all packages installed: pip install -r requirements.txt
```

## 📝 Example Queries the Chatbot Handles

✅ **Fees & Payments**
- "What are the tuition fees?"
- "Do you have scholarships?"

✅ **Exams**
- "When are midterm exams?"
- "What's the pass mark?"

✅ **Placements**
- "What's the average salary?"
- "Which companies visit campus?"

✅ **Facilities**
- "What are library timings?"
- "Do you have sports facilities?"

✅ **Admission**
- "How do I apply?"
- "What documents are needed?"

## 🚀 Deployment Options

### Local Development
```bash
streamlit run app.py
```

### Docker Deployment
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["streamlit", "run", "app.py"]
```

### Cloud Deployment (Streamlit Cloud, Azure, AWS, etc.)
1. Push repository to GitHub
2. Connect to Streamlit Cloud / Cloud Provider
3. Set environment variables (GROQ_API_KEY)
4. Deploy!

## 📦 Requirements

See `requirements.txt` for complete list:
- streamlit (UI framework)
- transformers (emotion detection)
- torch (deep learning backend)
- scikit-learn (intent classification)
- groq (Groq API client)
- requests (HTTP for Ollama)
- python-dotenv (environment variables)
- pandas (data handling)
- numpy (numerical operations)

## 🎓 Learning Resources

- **scikit-learn**: https://scikit-learn.org/
- **Transformers**: https://huggingface.co/transformers/
- **Streamlit**: https://streamlit.io/docs
- **Groq API**: https://console.groq.com/docs
- **Ollama**: https://ollama.ai

## 📄 License

This project is provided as-is for educational and commercial use.

## 👥 Support

For issues, questions, or improvements:
1. Check the troubleshooting section
2. Review code comments and docstrings
3. Examine debug panel info in Streamlit UI
4. Check logs via CLI: `python main.py --logs 50`

## 🎉 Future Enhancements

- [ ] Multi-language support
- [ ] Voice input/output
- [ ] Integration with actual college ERP systems
- [ ] Advanced RAG (Retrieval Augmented Generation)
- [ ] Fine-tuned models for domain-specific tasks
- [ ] REST API wrapper
- [ ] Mobile app integration
- [ ] Advanced analytics dashboard

---

**Made with ❤️ for college students and staff**
