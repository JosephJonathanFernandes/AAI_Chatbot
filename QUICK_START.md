# 🚀 Quick Start Guide

## ⚡ 5-Minute Setup

### Step 1: Get Groq API Key (2 minutes)
1. Go to https://console.groq.com
2. Sign up/Login
3. Create API key
4. Copy your key

### Step 2: Configure Project (1 minute)
```bash
# Copy example env file
copy .env.example .env

# Edit .env and add your Groq API key
GROQ_API_KEY=your_key_here
```

### Step 3: Install & Run (2 minutes)
```bash
# Install dependencies
pip install -r requirements.txt

# Run setup
python setup.py

# Start the chatbot
streamlit run app.py
```

**That's it! 🎉 Open http://localhost:8501**

---

## 📚 Project Files Guide

### Core Application
- **app.py** - Main Streamlit web interface (START HERE)
- **main.py** - Optional CLI mode for testing

### ML/AI Modules
- **intent_model.py** - Intent classification (what the user is asking)
- **emotion_detector.py** - Emotion detection (happy, stressed, confused, etc.)
- **llm_handler.py** - AI response generation (Groq API)
- **context_manager.py** - Conversation memory

### Infrastructure
- **database.py** - SQLite logging
- **utils.py** - Helper functions
- **requirements.txt** - Dependencies

### Data
- **data/intents.json** - Intent training examples
- **data/college_data.json** - College information (fees, exams, etc.)

---

## 🎯 How It Works (Architecture)

```
User Input
    ↓
Intent Classifier (scikit-learn)
    ↓ (Detects: fees, exams, placements, etc.)
Emotion Detector (Transformers/DistilBERT)
    ↓ (Detects: happy, stressed, confused, etc.)
Context Manager (Conversation History)
    ↓ (Tracks previous turns)
LLM Handler (Groq API or Ollama fallback)
    ↓ (Generates response with context)
Database Logger (SQLite)
    ↓ (Logs all interactions)
Streamlit UI
    ↓ (Displays to user)
```

---

## 💬 Example Usage

### Ask about tuition fees:
```
You: How much does engineering cost?
Assistant: Engineering costs $120,000 per year...
[Intent: fees | Confidence: 95% | Emotion: neutral]
```

### Ask about placements:
```
You: What's the average salary after graduation?
Assistant: The average starting salary is $450,000 per annum...
[Intent: placements | Confidence: 92% | Emotion: neutral]
```

### Query out of scope:
```
You: What's your favorite color?
Assistant: This is beyond my scope as a college assistant.
[Intent: unknown | Confidence: 28% | Emotion: neutral]
```

---

## 🔧 Common Tasks

### Train the Intent Model
```bash
python main.py --train
```

### View Recent Logs
```bash
python main.py --logs 50
```

### Use CLI Mode Instead of Streamlit
```bash
python main.py
# Then type your questions!
```

### Add New College Information
Edit `data/college_data.json` with your college details

### Add New Intent Types
Edit `data/intents.json` and retrain:
```bash
python main.py --train
```

---

## 📊 Debug Panel in Streamlit

Enable in the sidebar to see:
- **Intent**: What the user is asking about
- **Confidence**: How sure the model is (0-100%)
- **Emotion**: Detected user emotion
- **LLM Source**: Which API responded (Groq or Ollama)
- **Response Time**: How fast the response was

---

## ⚠️ Troubleshooting

### "API Key Not Found"
→ Make sure `.env` file has `GROQ_API_KEY=your_key_here`

### "Ollama not connecting"
→ Start Ollama first: `ollama serve` in another terminal

### "Transformer model download timeout"
→ Pre-download: `python -c "from transformers import pipeline; pipeline('sentiment-analysis')"`

### "Database locked"
→ Close all Streamlit tabs and restart

### "ModuleNotFoundError"
→ Install requirements: `pip install -r requirements.txt`

---

## 📈 Performance Tips

1. **First run is slower** - Models downloading/caching
2. **Use Groq API** - Much faster than local Ollama
3. **Clear old logs** - Keep database fast: `database.clear_logs(older_than_days=30)`
4. **Check statistics** - See what queries are popular

---

## 🎓 Key Features

✅ **Intent Classification** - Knows what students ask about  
✅ **Emotion Detection** - Responds empathetically  
✅ **Context Awareness** - Remembers conversation  
✅ **Time Awareness** - Says "Good morning/evening"  
✅ **Fallback Support** - Works without internet using Ollama  
✅ **Analytics** - See what students ask about  
✅ **Persistent Logging** - Every interaction saved  
✅ **Production Ready** - Error handling, caching, optimization  

---

## 📚 Learn More

- See **README.md** for detailed documentation
- Check **data/college_data.json** for available information
- Review module docstrings for code details
- Check **database.py** for available analytics

---

## 🎉 Ready to Launch

Your college chatbot is production-ready! Next:

1. ✅ Customize college data
2. ✅ Test with Streamlit UI
3. ✅ Deploy to a server
4. ✅ Share with students and staff

**Good luck! 🚀**
