# 🚀 News Recommendation API

A powerful FastAPI-based news recommendation service with machine learning capabilities, built for modern news applications.

## ✨ Features

- **🤖 ML-Powered Recommendations**: Smart content-based filtering using SentenceTransformers
- **🔐 JWT Authentication**: Secure user authentication with Google OAuth support
- **📊 User Analytics**: Track user preferences, likes, and reading patterns
- **🎲 Smart Fallbacks**: Diverse random recommendations for new users
- **⚡ High Performance**: Optimized for fast response times and scalability
- **🌐 CORS Enabled**: Ready for frontend integration

## 🛠️ Tech Stack

- **Backend**: FastAPI (Python 3.11+)
- **Database**: MongoDB with PyMongo
- **ML**: scikit-learn, pandas, SentenceTransformers
- **Auth**: JWT with Google OAuth2
- **Deployment**: Vercel-ready with optimized configuration

## 🚀 Quick Start

### Local Development

1. **Clone the repository**

   ```bash
   git clone <your-repo-url>
   cd python-service
   ```

2. **Create virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**

   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. **Run the server**

   ```bash
   uvicorn api:app --reload
   ```

6. **Visit API docs**
   ```
   http://localhost:8000/docs
   ```

### Production Deployment

See [README-DEPLOYMENT.md](README-DEPLOYMENT.md) for detailed Vercel deployment instructions.

## 📋 Environment Variables

```env
MONGO_URI=mongodb+srv://username:password@cluster.mongodb.net/
JWT_SECRET_KEY=your-super-secret-jwt-key-here
GOOGLE_CLIENT_ID=your-google-client-id.googleusercontent.com
DB_NAME=news_db_prod
```

## 🔗 API Endpoints

### Authentication

- `POST /register` - User registration
- `POST /token` - Login with email/password
- `POST /auth/google` - Google OAuth login

### Recommendations

- `GET /recommendations` - Get personalized recommendations
- `GET /trending` - Get trending articles

### User Actions

- `POST /like` - Like an article
- `DELETE /like` - Unlike an article
- `GET /likes` - Get user's liked articles
- `POST /seen` - Mark articles as seen

### Content

- `GET /browse` - Browse all articles with filters

## 🧠 ML Recommendation Engine

### Algorithm Features

- **Content-Based Filtering**: Uses SentenceTransformers for semantic similarity
- **Diverse Recommendations**: Intelligent tag diversification for new users
- **Adaptive Learning**: Learns from user preferences over time
- **Cold Start Handling**: Smart random recommendations with category diversity

### Performance

- **Model**: all-MiniLM-L6-v2 (384-dimensional embeddings)
- **Similarity**: Cosine similarity for content matching
- **Fallback**: Tag-diverse random selection for new users
- **Memory**: Optimized for 1GB+ serverless deployment

## 📊 Database Schema

### Collections

- **users**: User profiles and authentication
- **news**: Article content and metadata
- **likes**: User-article preference mapping
- **seen_articles**: Reading history tracking

### Indexes

- User email (unique)
- User-article likes (compound, unique)
- User-article seen (compound, unique)

## 🔧 Configuration Files

- `vercel.json` - Vercel deployment configuration
- `requirements.txt` - Python dependencies
- `runtime.txt` - Python version specification
- `.gitignore` - Git ignore patterns
- `deploy.sh` - Automated deployment script

## 🧪 Development

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx

# Run tests
pytest
```

### Code Quality

```bash
# Format code
black .

# Lint code
flake8 .

# Type checking
mypy .
```

## 📈 Performance Optimization

### For Production

- Use connection pooling for MongoDB
- Implement Redis caching for frequent requests
- Pre-compute embeddings for better performance
- Monitor with APM tools

### For Vercel

- Optimize cold start times
- Use Vercel Pro for better limits
- Implement proper error handling
- Monitor function metrics

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/) for the excellent web framework
- [SentenceTransformers](https://www.sbert.net/) for semantic text embeddings
- [Vercel](https://vercel.com/) for serverless deployment platform

---

**Built with ❤️ for the modern web**
