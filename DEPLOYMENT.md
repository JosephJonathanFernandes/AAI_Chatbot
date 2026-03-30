"""
Deployment Guide for College AI Assistant.
Production setup instructions and best practices.
"""

# # 🚀 Deployment Guide

## Deployment Options

### 1. Local Development (Recommended First)
```bash
# Simple local deployment
streamlit run app.py

# Access at: http://localhost:8501
```

### 2. Streamlit Cloud (Easiest)

1. Push code to GitHub
2. Go to https://streamlit.io/cloud
3. Click "New app"
4. Select your repository and main file (app.py)
5. Set environment variables in Secrets

**Secrets Configuration:**
```toml
# .streamlit/secrets.toml (or through UI)
GROQ_API_KEY = "your_api_key_here"
OLLAMA_BASE_URL = "http://localhost:11434/api/generate"
```

### 3. Docker Deployment

**Dockerfile:**
```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy files
COPY requirements.txt .
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download transformer model
RUN python -c "from transformers import pipeline; pipeline('sentiment-analysis')"

# Create directory for data
RUN mkdir -p data

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

**Build and run:**
```bash
docker build -t college-chatbot .
docker run -p 8501:8501 \
  -e GROQ_API_KEY=your_key \
  -v $(pwd)/data:/app/data \
  college-chatbot
```

### 4. Cloud Platforms

#### Azure App Service
```bash
# Create resource group
az group create --name college-chatbot --location eastus

# Create App Service Plan
az appservice plan create \
  --name chatbot-plan \
  --resource-group college-chatbot \
  --sku B2 \
  --is-linux

# Create web app
az webapp create \
  --resource-group college-chatbot \
  --plan chatbot-plan \
  --name college-chatbot-app \
  --runtime "PYTHON|3.10"

# Set environment variables
az webapp config appsettings set \
  --resource-group college-chatbot \
  --name college-chatbot-app \
  --settings GROQ_API_KEY=your_key_here

# Deploy
git push azure main
```

#### AWS
```bash
# Using Elastic Beanstalk
eb init -p python-3.10 college-chatbot
eb create college-chatbot-env
eb deploy

# Set environment variable
eb setenv GROQ_API_KEY=your_key_here
```

#### Google Cloud Run
```bash
# Build and deploy
gcloud run deploy college-chatbot \
  --source . \
  --platform managed \
  --region us-central1 \
  --set-env-vars GROQ_API_KEY=your_key_here \
  --allow-unauthenticated
```

### 5. On-Premise Server (VPS/Dedicated)

```bash
# SSH into server
ssh user@your_server_ip

# Install dependencies
sudo apt-get update
sudo apt-get install python3.10 python3-pip screen git

# Clone repository
git clone https://github.com/your-repo/college-chatbot.git
cd college-chatbot

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Create .env file
cat > .env << EOF
GROQ_API_KEY=your_key_here
EOF

# Run with screen (keeps running after disconnect)
screen -S chatbot
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
# Press Ctrl+A then D to detach

# Access at: http://your_server_ip:8501
```

## Production Best Practices

### 1. Security
- [ ] Use HTTPS/SSL certificates (Let's Encrypt)
- [ ] Set `GROQ_API_KEY` as environment variable (never commit)
- [ ] Enable authentication if needed (Streamlit Authenticator)
- [ ] Use firewall rules (whitelist IPs)
- [ ] Regular security updates

### 2. Performance
- [ ] Use Redis for caching
- [ ] Database connection pooling
- [ ] Load balancing (multiple instances)
- [ ] CDN for static files
- [ ] Monitor response times

### 3. Monitoring
- [ ] Set up logging aggregation (ELK, Datadog)
- [ ] Error tracking (Sentry)
- [ ] Performance monitoring (APM)
- [ ] Uptime monitoring (UptimeRobot)
- [ ] Database monitoring

### 4. Scalability
- [ ] Horizontal scaling with load balancer
- [ ] Database read replicas
- [ ] Session management (Redis/sticky sessions)
- [ ] Async task queue for heavy operations
- [ ] Database optimization/indexing

### 5. Maintenance
- [ ] Regular backups (database, models)
- [ ] Log rotation
- [ ] Model retraining schedule
- [ ] Dependency updates
- [ ] Performance profiling

## Configuration for Production

### `streamlit_config.toml`
```toml
[theme]
primaryColor = "#2196F3"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F5F5F5"
textColor = "#000000"

[client]
showErrorDetails = false
toolbarMode = "minimal"

[server]
maxUploadSize = 200
enableXsrfProtection = true
port = 8501
headless = true
runOnSave = false
```

### Environment Variables
```env
# Production critical
GROQ_API_KEY=your_production_key
ENVIRONMENT=production

# Database
DATABASE_PATH=/var/data/chatbot.db
DATABASE_BACKUP_PATH=/var/backups/

# Logging
LOG_LEVEL=INFO
LOG_FILE=/var/logs/chatbot.log

# Performance
CACHE_ENABLED=true
MAX_WORKERS=4
```

## Monitoring & Alerts

### Key Metrics to Monitor
- Response time (target: < 2s)
- Error rate (target: < 0.1%)
- API fallback rate (target: < 5%)
- Database query time (target: < 100ms)
- User session count
- Intent distribution

### Sample Monitoring Script
```python
from database import ChatbotDatabase

def check_health():
    db = ChatbotDatabase()
    analytics = db.get_analytics_summary()
    
    # Alert if fallback rate too high
    if analytics['fallback_rate'] > 0.1:
        alert("High Ollama fallback rate!")
    
    # Alert if no recent interactions
    if analytics['total_interactions'] == 0:
        alert("No interactions detected!")
    
    return analytics
```

## Disaster Recovery

### Backup Strategy
```bash
# Daily database backup
0 2 * * * cp /path/to/chatbot.db /backups/chatbot_$(date +%Y%m%d).db

# Keep 30 days of backups
find /backups -name "chatbot_*.db" -mtime +30 -delete
```

### Recovery Procedure
1. Stop the application
2. Restore database from backup
3. Verify data integrity
4. Restart application
5. Test basic functionality

## Performance Optimization

### Database Optimization
```python
# Create indexes
from database import ChatbotDatabase
db = ChatbotDatabase()
conn = db.get_connection()
cursor = conn.cursor()

# Index frequently queried columns
cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON logs(timestamp)")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_intent ON logs(intent)")
cursor.execute("VACUUM")
conn.commit()
```

### API Rate Limiting
```python
from functools import wraps
import time

def rate_limit(max_per_minute=60):
    def decorator(f):
        calls = []
        
        @wraps(f)
        def wrapper(*args, **kwargs):
            now = time.time()
            calls[:] = [c for c in calls if c > now - 60]
            
            if len(calls) >= max_per_minute:
                raise Exception("Rate limit exceeded")
            
            calls.append(now)
            return f(*args, **kwargs)
        
        return wrapper
    return decorator
```

## Load Testing

```bash
# Install locust
pip install locust

# Create locustfile.py and run
locust -f locustfile.py --host=http://localhost:8501
```

## Rollback Strategy

If deployment goes wrong:

```bash
# Keep previous version
mv current/ current.old/

# Restore previous version
mv backups/previous_version/ current/

# Restart service
systemctl restart chatbot

# Test
curl http://localhost:8501
```

## CI/CD Pipeline

### GitHub Actions Example
```yaml
name: Deploy

on:
  push:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Run tests
      run: python -m pytest
    
    - name: Deploy to production
      env:
        DEPLOY_KEY: ${{ secrets.DEPLOY_KEY }}
      run: |
        ssh -i $DEPLOY_KEY user@server 'cd /app && git pull && systemctl restart chatbot'
```

## Support & Maintenance

- Monitor logs daily
- Review analytics weekly
- Update models monthly
- Security audit quarterly
- Plan for scaling annually

---

**Production deployment readiness: ✅ Complete**
