# PowerShell script to setup and run the Resume Relevance Check System

Write-Host "🚀 Resume Relevance Check System - Setup & Launch" -ForegroundColor Green
Write-Host "=" * 50

# Set the API key as environment variable
$apiKey = "AIzaSyDAx61-09OGYB0J6ab2BvgPI3ZIHM7MTYg"

# Set for current session
$env:GEMINI_API_KEY = $apiKey

# Set permanently for current user (optional)
try {
    [Environment]::SetEnvironmentVariable("GEMINI_API_KEY", $apiKey, "User")
    Write-Host "✅ GEMINI_API_KEY environment variable has been set!" -ForegroundColor Green
} catch {
    Write-Host "⚠️ Could not set permanent environment variable, using session variable" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "📦 Installing dependencies..." -ForegroundColor Blue

# Install requirements
try {
    pip install -r requirements.txt
    Write-Host "✅ Dependencies installed successfully!" -ForegroundColor Green
} catch {
    Write-Host "❌ Failed to install dependencies. Please run 'pip install -r requirements.txt' manually" -ForegroundColor Red
    Read-Host "Press Enter to continue anyway"
}

Write-Host ""
Write-Host "🚀 Launching application..." -ForegroundColor Blue
Write-Host "The application will open in your default browser"
Write-Host ""

# Launch the application
try {
    streamlit run app.py
} catch {
    Write-Host "❌ Failed to launch application" -ForegroundColor Red
    Write-Host "Please run 'streamlit run app.py' manually" -ForegroundColor Yellow
}

Read-Host "Press Enter to exit"