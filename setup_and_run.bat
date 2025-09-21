@echo off
echo Setting up Gemini API Key Environment Variable
echo.

REM Set the API key as environment variable for current session
set GEMINI_API_KEY=AIzaSyDAx61-09OGYB0J6ab2BvgPI3ZIHM7MTYg

REM Set the API key permanently for current user (optional)
setx GEMINI_API_KEY "AIzaSyDAx61-09OGYB0J6ab2BvgPI3ZIHM7MTYg"

echo ✅ GEMINI_API_KEY environment variable has been set!
echo.
echo Now launching the Resume Relevance Check System...
echo.

REM Install requirements if needed
pip install -r requirements.txt

REM Launch the application
streamlit run app.py

pause