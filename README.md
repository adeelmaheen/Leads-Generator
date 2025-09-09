# Installation and Setup Instructions

## Quick Start

1. **Clone or download the project files**
   
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up API keys:**
   - Copy `.env.example` to `.env`
   - Get your Exa API key from https://exa.ai
   - Get your OpenAI API key from https://platform.openai.com
   - Add the keys to your `.env` file:
   ```
   EXA_API_KEY=your_actual_exa_key
   OPENAI_API_KEY=your_actual_openai_key
   ```

4. **Run the application:**
   ```bash
   streamlit run leadfinder_ai.py
   ```

5. **Open your browser** to the URL shown (usually http://localhost:8501)

## Alternative Installation with setup.py

```bash
pip install -e .
```

## Usage

1. Enter a topic or expertise area (e.g., "blockchain security", "sustainable packaging")
2. Adjust settings in the sidebar:
   - Number of leads to find
   - Search type (auto, neural, keyword, fast)
   - Enable/disable LinkedIn search
   - Enable/disable full-text analysis
3. Click "Find Leads" to start the search
4. View results in the table and download as CSV if needed

## API Key Setup

### Exa API Key
1. Go to https://exa.ai and sign up for an account
2. Navigate to your dashboard and create an API key
3. Add it to your `.env` file as `EXA_API_KEY=your_key_here`

### OpenAI API Key
1. Go to https://platform.openai.com and sign up/log in
2. Navigate to API keys section and create a new key
3. Add it to your `.env` file as `OPENAI_API_KEY=your_key_here`

## Project Structure
```
leadfinder-ai/
├── leadfinder_ai.py      # Main application file
├── requirements.txt      # Python dependencies
├── .env.example         # Environment variables template
├── .env                 # Your actual API keys (create this)
├── setup.py            # Package setup file
└── README.md           # Project documentation
```

## Features

- **AI-Powered Search**: Uses Exa's neural search engine for relevant results
- **LinkedIn Integration**: Finds professional profiles (simulated in this version)
- **Contact Extraction**: Parses phone numbers and emails from web content
- **AI Ranking**: Uses OpenAI to score and rank leads by relevance
- **Interactive UI**: Clean Streamlit interface with real-time progress
- **Export Functionality**: Download results as CSV for further use

## Troubleshooting

### Common Issues

1. **"Module not found" errors**
   - Make sure you've installed all requirements: `pip install -r requirements.txt`

2. **API key errors**
   - Verify your keys are correct in the `.env` file
   - Make sure the `.env` file is in the same directory as the Python script
   - Check that you have sufficient API credits

3. **No results found**
   - Try different search terms or topics
   - Adjust the search type (try "auto" for best results)
   - Increase the number of leads to search for

4. **LinkedIn search not working**
   - The current implementation simulates LinkedIn search using regular web search
   - For full LinkedIn integration, you'd need to set up the Exa MCP server

### Performance Tips

- Use "fast" search type for quicker results during testing
- Start with smaller numbers of leads (5-10) to test functionality
- Enable full-text analysis only when you need contact information extraction

## Next Steps

To fully implement the original vision, consider:

1. **Set up Exa MCP Server** for true LinkedIn search capabilities
2. **Implement OpenAI Agents SDK** for more sophisticated agent orchestration
3. **Add more sophisticated contact parsing** using NLP models
4. **Integrate with CRM systems** like Salesforce or HubSpot
5. **Add caching** to avoid repeated API calls for the same searches