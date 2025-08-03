# 🚀 Deploy Your ML-Powered News API to Vercel

Your FastAPI service includes a complete ML recommendation system with:

- **SentenceTransformers** for semantic similarity
- **Scikit-learn** for recommendation algorithms
- **Pandas** for data processing
- **Smart recommendation engine** with diverse random fallbacks

## 📋 Pre-Deployment Checklist

✅ **MongoDB Database** - Make sure your MongoDB instance is accessible from Vercel
✅ **Environment Variables** - Prepare your secrets (see below)
✅ **Vercel Account** - Free tier works, Pro recommended for better performance

## 🏃‍♂️ Quick Start

### 1. Install Vercel CLI

```bash
npm install -g vercel
```

### 2. Deploy

```bash
cd python-service
./deploy.sh  # or follow manual steps below
```

### 3. Set Environment Variables

In your Vercel dashboard, add:

```env
MONGO_URI=mongodb+srv://username:password@cluster.mongodb.net/
JWT_SECRET_KEY=your-super-secret-jwt-key-here
GOOGLE_CLIENT_ID=your-google-client-id.googleusercontent.com
DB_NAME=news_db_prod
```

## 🛠️ Manual Deployment

```bash
# 1. Login to Vercel
vercel login

# 2. Deploy (first time will take 5-10 minutes)
vercel --prod

# 3. Follow the prompts and note your deployment URL
```

## ⚡ Performance Expectations

| Metric               | Value     | Notes                             |
| -------------------- | --------- | --------------------------------- |
| **First Deployment** | 5-10 min  | Installing ML libraries           |
| **Cold Start**       | 15-30s    | Loading SentenceTransformer model |
| **Warm Response**    | 100-500ms | Once model is loaded              |
| **Memory Usage**     | ~800MB    | Optimized for 1024MB limit        |
| **Function Timeout** | 60s       | Configured for model loading      |

## 🔧 Configuration Details

### `vercel.json` Features:

- **Memory**: 1024MB allocation for ML models
- **Timeout**: 60 seconds for model initialization
- **Runtime**: Python 3.11.x

### `requirements.txt` Optimizations:

- ✅ Full ML stack included
- ✅ `googletrans` commented out (optional)
- ✅ Latest stable versions

## 🚨 Troubleshooting

### Deployment Issues

**❌ "Function size limit exceeded"**

```bash
# Solution: Upgrade to Vercel Pro or remove googletrans
# googletrans is already commented out in requirements.txt
```

**❌ "Build timeout"**

```bash
# Solution: This is normal for first deployment with ML libraries
# Wait up to 10 minutes, then try again if it fails
```

### Runtime Issues

**❌ "Function timeout on first request"**

```bash
# Expected: SentenceTransformer model loading takes time
# Wait 30-60 seconds, subsequent requests will be fast
```

**❌ "Memory limit exceeded"**

```bash
# Solution: Upgrade to Vercel Pro for 3GB memory limit
```

## 🎯 Testing Your Deployment

Once deployed, test these endpoints:

```bash
# Health check
curl https://your-app.vercel.app/

# API documentation
open https://your-app.vercel.app/docs

# Test recommendations (will be slow on first call)
curl -H "Authorization: Bearer YOUR_JWT_TOKEN" \
     https://your-app.vercel.app/recommendations
```

## 📈 Optimization Tips

### For Better Performance:

1. **Upgrade to Vercel Pro** - 3GB memory, longer timeouts
2. **Implement warmup** - Add a simple warmup endpoint
3. **Cache embeddings** - Pre-compute and store in database
4. **Monitor usage** - Watch Vercel function logs

### For Cost Optimization:

1. **Use caching** - Implement Redis for frequent requests
2. **Optimize queries** - Add database indexes
3. **Batch processing** - Process multiple recommendations together

## 🔄 Updates & Maintenance

```bash
# Update your deployment
git push origin main  # If connected to Git repo
# OR
vercel --prod         # Manual redeploy
```

## 📊 Monitoring

Monitor your deployment:

- **Vercel Dashboard** - Function logs and metrics
- **MongoDB Atlas** - Database performance
- **Response times** - Cold start vs warm performance

Your ML-powered news recommendation API is now live! 🎉
