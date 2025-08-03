# Vercel Deployment Guide

This guide will help you deploy your Python FastAPI service to Vercel.

## 🚀 Quick Deploy

### 1. Install Vercel CLI

```bash
npm i -g vercel
```

### 2. Login to Vercel

```bash
vercel login
```

### 3. Deploy

```bash
# From the python-service directory
vercel --prod
```

## ⚙️ Configuration Files

### `vercel.json`

- Configures Vercel to use Python runtime
- Sets up routing to your FastAPI app
- Configures function timeout

### `runtime.txt`

- Specifies Python 3.11.x for deployment

### `requirements-vercel.txt`

- Optimized dependencies for Vercel's size limits
- Removes heavy ML libraries to avoid deployment issues

## 🔧 Required Environment Variables

Set these in your Vercel dashboard (Project Settings → Environment Variables):

```env
MONGO_URI=mongodb+srv://username:password@cluster.mongodb.net/
JWT_SECRET_KEY=your-super-secret-jwt-key-here
GOOGLE_CLIENT_ID=your-google-client-id.googleusercontent.com
DB_NAME=news_db_prod
NODE_ENV=production
```

## 📦 Package Optimization

### For Standard Deployment:

1. Rename `requirements-vercel.txt` to `requirements.txt`:
   ```bash
   cp requirements-vercel.txt requirements.txt
   ```

### For Full ML Features:

If you need the ML recommendation features:

1. **Option A: Vercel Pro** (Recommended)

   - Upgrade to Vercel Pro for larger function limits
   - Use the original `requirements.txt`

2. **Option B: Hybrid Architecture**

   - Deploy API without ML features to Vercel
   - Deploy ML service separately (Railway, Render, etc.)
   - Call ML service from main API

3. **Option C: Pre-computed Embeddings**
   - Pre-compute article embeddings
   - Store in database
   - Use lighter similarity calculations

## 🛠️ Deployment Steps

1. **Prepare your repository:**

   ```bash
   # Copy optimized requirements
   cp requirements-vercel.txt requirements.txt

   # Commit changes
   git add .
   git commit -m "Add Vercel deployment config"
   ```

2. **Deploy to Vercel:**

   ```bash
   vercel --prod
   ```

3. **Set environment variables** in Vercel dashboard

4. **Test your deployment** at the provided URL

## 🔍 API Endpoints

After deployment, your API will be available at:

- `https://your-app.vercel.app/docs` - API documentation
- `https://your-app.vercel.app/health` - Health check (if implemented)

## 🚨 Common Issues

### 1. Function Size Limit

- **Problem**: Deployment fails due to package size
- **Solution**: Use `requirements-vercel.txt` or upgrade to Vercel Pro

### 2. Cold Start Timeouts

- **Problem**: First request takes too long
- **Solution**: Implement proper error handling and consider Vercel Pro

### 3. Database Connection

- **Problem**: Can't connect to MongoDB
- **Solution**: Ensure MONGO_URI is correctly set and database allows connections

## 📚 Alternative Platforms

If Vercel doesn't meet your needs:

- **Railway** - Great for full-stack Python apps
- **Render** - Free tier with good Python support
- **Heroku** - Classic choice for Python deployments
- **DigitalOcean App Platform** - Good performance and pricing

## 🔄 Updates

To update your deployment:

```bash
git push origin main  # If connected to Git
# OR
vercel --prod         # For manual deployments
```
