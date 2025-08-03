#!/bin/bash

# Vercel Deployment Script for Python Service

echo "🚀 Preparing for Vercel deployment..."

# Check if we're in the right directory
if [ ! -f "api.py" ]; then
    echo "❌ Error: api.py not found. Make sure you're in the python-service directory."
    exit 1
fi

# Install Vercel CLI if not present
if ! command -v vercel &> /dev/null; then
    echo "📦 Installing Vercel CLI..."
    npm install -g vercel
fi

# Using full ML requirements for deployment
echo "📋 Using full ML requirements.txt for deployment..."

# Show current files that will be deployed
echo "📁 Files to be deployed:"
ls -la | grep -E "\.(py|txt|json|md)$"

echo ""
echo "⚙️  Make sure you have set these environment variables in Vercel:"
echo "   - MONGO_URI"
echo "   - JWT_SECRET_KEY" 
echo "   - GOOGLE_CLIENT_ID"
echo "   - DB_NAME (optional)"
echo ""

# Login to Vercel
echo "🔑 Logging in to Vercel (if needed)..."
vercel login

# Deploy to Vercel
echo "🚀 Deploying to Vercel..."
vercel --prod

echo "✅ Deployment complete!"
echo ""
echo "📝 Next steps:"
echo "1. Set environment variables in Vercel dashboard"
echo "2. Test your API at the provided URL"
echo "3. Check logs if there are any issues"
echo ""
echo "🔗 Useful commands:"
echo "   vercel logs <deployment-url>   # Check deployment logs"
echo "   vercel env ls                  # List environment variables"
echo "   vercel --prod                  # Redeploy"