# Resume Relevance Check System - Setup and Usage Guide

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Google Gemini API key (automatically configured)

### Easy Setup Options

**Option 1: Automated Setup (Recommended)**
```bash
# For Windows Command Prompt
setup_and_run.bat

# For Windows PowerShell
setup_and_run.ps1

# For Python
python launch.py
```

**Option 2: Manual Setup**
```bash
pip install -r requirements.txt
streamlit run app.py
```

### API Key Configuration

The application comes with a pre-configured API key for immediate use. For production or personal use, you can:

1. **Use Environment Variable (Recommended for security):**
   ```bash
   # Windows Command Prompt
   set GEMINI_API_KEY=your_api_key_here
   
   # Windows PowerShell
   $env:GEMINI_API_KEY="your_api_key_here"
   
   # Linux/Mac
   export GEMINI_API_KEY=your_api_key_here
   ```

2. **Update the default key in the code:**
   - Edit `app.py` and update `DEFAULT_GEMINI_API_KEY` variable

3. **Get your own API key:**
   - Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Sign in and create a new API key
   - Replace the default key with your own

## ✨ New Features & Improvements

### 🔧 Enhanced Error Handling
- Comprehensive exception handling throughout the application
- User-friendly error messages with actionable solutions
- Graceful degradation when dependencies are missing
- Detailed logging for debugging

### 🛡️ Input Validation & Security
- File size and type validation (PDF/DOCX only, max 10MB)
- Text input validation with length constraints
- API key validation and secure handling
- Filename sanitization for safe storage
- Database integrity constraints

### ⚡ Performance Optimizations
- Intelligent caching for API responses (1 hour TTL)
- Database query caching (5-10 minutes TTL)
- Optimized file processing
- Progress indicators for long operations
- Reduced API calls through smart caching

### 🎨 Enhanced User Interface
- **5 Main Tabs**: Job Management, Resume Evaluation, Dashboard, Analytics, Settings
- **Progress Tracking**: Real-time progress bars for batch operations
- **Interactive Filters**: Score-based filtering, verdict filtering, search functionality
- **Enhanced Metrics**: Comprehensive scoring breakdown and insights
- **Responsive Design**: Better mobile and desktop experience

### 📊 Advanced Analytics
- **Score Distribution Analysis**: Detailed breakdown of candidate performance
- **Top Performers Dashboard**: Ranked candidate lists with filtering
- **System-wide Insights**: Performance trends and statistics
- **Export Capabilities**: CSV exports for HR teams and reporting

### 🔄 Batch Processing
- **Multi-file Upload**: Process multiple resumes simultaneously
- **Progress Tracking**: Real-time status for each file
- **Error Handling**: Continue processing even if individual files fail
- **Results Summary**: Comprehensive batch analysis results

### ⚙️ Configuration Management
- **Settings Panel**: Customizable file size limits and processing options
- **Cache Management**: Clear cache functionality
- **Data Export/Import**: Backup and restore capabilities
- **System Health**: Dependency and database status monitoring

## 📋 Feature Comparison

| Feature | Original | Enhanced |
|---------|----------|----------|
| Error Handling | Basic | Comprehensive with logging |
| File Validation | Minimal | Size, type, content validation |
| Caching | None | Multi-level intelligent caching |
| UI/UX | Simple tabs | Interactive dashboard with filters |
| Analytics | Basic stats | Advanced insights & trends |
| Batch Processing | Sequential | Parallel with progress tracking |
| Export Options | CSV only | Multiple formats with templates |
| Security | Basic | Input validation & sanitization |
| Database | Simple schema | Enhanced with constraints & indexes |
| Settings | None | Comprehensive configuration panel |

## 🏗️ Technical Improvements

### Database Enhancements
- **Foreign Key Constraints**: Data integrity enforcement
- **Indexes**: Improved query performance
- **Enhanced Schema**: Additional fields for better analytics
- **Validation Constraints**: Data quality enforcement

### API Integration
- **Response Validation**: JSON structure validation
- **Retry Logic**: Automatic retry for failed requests
- **Rate Limiting**: Intelligent API usage management
- **Error Recovery**: Graceful handling of API failures

### File Processing
- **Enhanced PDF Extraction**: Better text extraction from complex PDFs
- **DOCX Table Support**: Extract text from tables and complex layouts
- **File Type Detection**: Accurate file format validation
- **Size Optimization**: Efficient memory usage for large files

## 📊 Usage Scenarios

### For HR Teams
1. **Bulk Resume Screening**: Upload 50+ resumes for quick initial screening
2. **Candidate Ranking**: Automatic scoring and ranking of applications
3. **Skills Gap Analysis**: Identify missing skills across candidate pool
4. **Reporting**: Export results for stakeholder reviews

### For Recruiters
1. **Job-Resume Matching**: Compare candidates against specific job requirements
2. **Quick Filtering**: Filter candidates by score thresholds
3. **Detailed Analysis**: View comprehensive candidate assessments
4. **Progress Tracking**: Monitor evaluation progress in real-time

### For Hiring Managers
1. **Dashboard Overview**: Get quick insights into candidate pool quality
2. **Top Candidates**: Focus on highest-scoring candidates first
3. **Skills Assessment**: Understand candidate strengths and gaps
4. **Data Export**: Share results with interview panels

## 🔧 Configuration Options

### API Settings
- **Gemini API Key**: Required for AI-powered analysis
- **Request Timeout**: Configurable timeout for API calls
- **Retry Attempts**: Number of retry attempts for failed requests

### Processing Settings
- **Max File Size**: Adjustable file size limits (1-50MB)
- **Batch Size**: Number of files to process simultaneously
- **Cache Duration**: How long to cache API responses

### Display Settings
- **Results Per Page**: Number of results to show
- **Score Thresholds**: Custom scoring ranges
- **Export Format**: CSV, Excel, or JSON

## 🚀 Performance Tips

1. **Use Environment Variables**: Set GEMINI_API_KEY to avoid re-entering
2. **Enable Caching**: Keep caching enabled for better performance
3. **Batch Processing**: Process multiple resumes together for efficiency
4. **Regular Cleanup**: Clear cache periodically to free memory
5. **File Optimization**: Use text-searchable PDFs for better extraction

## 🐛 Troubleshooting

### Common Issues

**API Key Errors:**
- Ensure your Gemini API key is valid and has quota
- Check if the key has the necessary permissions

**File Processing Errors:**
- Verify file format (PDF/DOCX only)
- Check file size limits
- Ensure files contain extractable text (not scanned images)

**Performance Issues:**
- Clear application cache
- Reduce batch size for large file processing
- Check internet connection for API calls

**Database Errors:**
- Restart the application to recreate database
- Check disk space availability
- Verify write permissions in the data folder

## 📈 Future Enhancements

- **Advanced Skill Matching**: Industry-specific skill taxonomies
- **Integration APIs**: Connect with HR systems and ATS platforms
- **Custom Scoring Models**: User-defined scoring weights
- **Email Notifications**: Automated alerts for high-match candidates
- **Historical Analytics**: Trend analysis over time
- **Multi-language Support**: Support for non-English resumes

## 📞 Support

For technical support or feature requests, please check the application's Settings tab for detailed help documentation and troubleshooting guides.