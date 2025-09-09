import streamlit as st
import pandas as pd
import os
import re
import json
import requests
from typing import List, Dict, Optional
from dotenv import load_dotenv
from exa_py import Exa
from openai import OpenAI
import time

# Load environment variables
load_dotenv()
EXA_API_KEY = os.getenv("EXA_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize clients
exa = Exa(api_key=EXA_API_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

class LeadFinderTools:
    """Collection of tools for finding and processing leads"""
    
    def __init__(self):
        self.exa = exa
        self.openai_client = openai_client
    
    def search_exa(self, query: str, search_type: str = "auto", text: bool = True, n: int = 10) -> List[Dict]:
        """
        Search using Exa's API with different search types
        
        Args:
            query: Search query/topic
            search_type: auto, neural, keyword, or fast
            text: Whether to retrieve full text content
            n: Number of results to return
        
        Returns:
            List of search results with metadata
        """
        try:
            results = self.exa.search_and_contents(
                query=query,
                type=search_type,
                use_autoprompt=True,
                num_results=n,
                text=text,
                summary=True,
                highlights=True
            )
            
            processed_results = []
            for result in results.results:
                processed_result = {
                    'title': result.title,
                    'url': result.url,
                    'author': getattr(result, 'author', ''),
                    'published_date': getattr(result, 'published_date', ''),
                    'summary': getattr(result, 'summary', ''),
                    'text': getattr(result, 'text', ''),
                    'highlights': getattr(result, 'highlights', [])
                }
                processed_results.append(processed_result)
            
            return processed_results
            
        except Exception as e:
            st.error(f"Error in Exa search: {str(e)}")
            return []
    
    def linkedin_search(self, query: str) -> List[Dict]:
        """
        Search LinkedIn using Exa's MCP tools (simulated for this implementation)
        Note: This would normally call the Exa MCP server's linkedin_search endpoint
        """
        try:
            # In a real implementation, this would call the MCP server
            # For now, we'll simulate LinkedIn search using regular Exa search
            linkedin_query = f"site:linkedin.com/in {query}"
            results = self.search_exa(linkedin_query, search_type="keyword", n=5)
            
            linkedin_profiles = []
            for result in results:
                if 'linkedin.com/in' in result['url']:
                    profile = {
                        'name': self.extract_name_from_linkedin(result),
                        'profile_url': result['url'],
                        'title': result['title'],
                        'summary': result['summary']
                    }
                    linkedin_profiles.append(profile)
            
            return linkedin_profiles
            
        except Exception as e:
            st.error(f"Error in LinkedIn search: {str(e)}")
            return []
    
    def extract_name_from_linkedin(self, result: Dict) -> str:
        """Extract name from LinkedIn profile data"""
        title = result.get('title', '')
        # LinkedIn titles are usually in format "Name | Title at Company"
        if ' | ' in title:
            return title.split(' | ')[0].strip()
        elif ' - ' in title:
            return title.split(' - ')[0].strip()
        return title
    
    def parse_contacts(self, text: str) -> Dict:
        """
        Parse contact information from text content
        
        Args:
            text: Full text content to parse
            
        Returns:
            Dictionary with extracted contact info
        """
        contacts = {
            'phones': [],
            'emails': [],
            'names': []
        }
        
        if not text:
            return contacts
        
        # Phone number patterns (international and US formats)
        phone_patterns = [
            r'\+\d{1,3}[-.\s]?\(?\d{1,3}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}',
            r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
            r'\d{3}[-.\s]?\d{3}[-.\s]?\d{4}'
        ]
        
        for pattern in phone_patterns:
            phones = re.findall(pattern, text)
            contacts['phones'].extend(phones)
        
        # Email patterns
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        contacts['emails'].extend(emails)
        
        # Simple name extraction from headers and strong tags
        name_patterns = [
            r'<h[1-6][^>]*>([^<]+)</h[1-6]>',
            r'<strong>([^<]+)</strong>',
            r'About\s+([A-Z][a-z]+\s+[A-Z][a-z]+)',
        ]
        
        for pattern in name_patterns:
            names = re.findall(pattern, text, re.IGNORECASE)
            contacts['names'].extend(names)
        
        # Remove duplicates
        contacts['phones'] = list(set(contacts['phones']))
        contacts['emails'] = list(set(contacts['emails']))
        contacts['names'] = list(set(contacts['names']))
        
        return contacts
    
    def build_lead(self, search_result: Dict, linkedin_data: Dict = None, contacts: Dict = None) -> Dict:
        """
        Build a structured lead from various data sources
        
        Args:
            search_result: Result from Exa search
            linkedin_data: LinkedIn profile data
            contacts: Extracted contact information
            
        Returns:
            Structured lead dictionary
        """
        lead = {
            'name': '',
            'linkedin_url': '',
            'phone': '',
            'email': '',
            'company_role': '',
            'description': '',
            'source_url': search_result.get('url', ''),
            'relevance_score': 0.8  # Default relevance
        }
        
        # Use LinkedIn data if available
        if linkedin_data:
            lead['name'] = linkedin_data.get('name', '')
            lead['linkedin_url'] = linkedin_data.get('profile_url', '')
            lead['company_role'] = linkedin_data.get('title', '')
            lead['description'] = linkedin_data.get('summary', '')
        else:
            # Fall back to search result data
            lead['name'] = search_result.get('author', '')
            lead['description'] = search_result.get('summary', '')
        
        # Add contact information if available
        if contacts:
            lead['phone'] = contacts['phones'][0] if contacts['phones'] else ''
            lead['email'] = contacts['emails'][0] if contacts['emails'] else ''
            if not lead['name'] and contacts['names']:
                lead['name'] = contacts['names'][0]
        
        # If no name found, try to extract from title
        if not lead['name']:
            title = search_result.get('title', '')
            if title:
                # Simple heuristic to extract name from title
                words = title.split()
                if len(words) >= 2:
                    potential_name = f"{words[0]} {words[1]}"
                    if re.match(r'^[A-Z][a-z]+\s+[A-Z][a-z]+$', potential_name):
                        lead['name'] = potential_name
        
        return lead

class LeadFinderAgent:
    """AI Agent that orchestrates the lead finding process"""
    
    def __init__(self):
        self.tools = LeadFinderTools()
        self.client = openai_client
    
    def find_leads(self, topic: str, num_leads: int = 10, search_type: str = "auto", 
                   use_linkedin: bool = True, use_full_text: bool = True) -> List[Dict]:
        """
        Main method to find leads for a given topic
        
        Args:
            topic: The subject matter or expertise area to search for
            num_leads: Maximum number of leads to return
            search_type: Exa search type (auto, neural, keyword, fast)
            use_linkedin: Whether to include LinkedIn search
            use_full_text: Whether to retrieve full text for contact parsing
            
        Returns:
            List of structured lead dictionaries
        """
        leads = []
        
        try:
            # Step 1: Search for relevant content
            st.write("🔍 Searching for relevant content...")
            search_results = self.tools.search_exa(
                query=f"expert {topic} specialist consultant professional",
                search_type=search_type,
                text=use_full_text,
                n=num_leads * 2  # Search for more to filter later
            )
            
            if not search_results:
                st.warning("No search results found. Please try a different topic.")
                return leads
            
            # Step 2: LinkedIn search if enabled
            linkedin_profiles = []
            if use_linkedin:
                st.write("👔 Searching LinkedIn profiles...")
                linkedin_profiles = self.tools.linkedin_search(topic)
            
            # Step 3: Process results and extract leads
            st.write("⚙️ Processing results and extracting contact information...")
            
            for i, result in enumerate(search_results[:num_leads]):
                # Parse contacts from full text if available
                contacts = None
                if use_full_text and result.get('text'):
                    contacts = self.tools.parse_contacts(result['text'])
                
                # Try to match with LinkedIn data
                linkedin_match = None
                if linkedin_profiles:
                    # Simple matching by checking if names appear in the search result
                    for profile in linkedin_profiles:
                        if profile['name'].lower() in result.get('text', '').lower():
                            linkedin_match = profile
                            break
                
                # Build the lead
                lead = self.tools.build_lead(result, linkedin_match, contacts)
                
                if lead['name']:  # Only add leads with names
                    leads.append(lead)
                
                # Progress update
                progress = (i + 1) / min(len(search_results), num_leads)
                st.progress(progress)
            
            # Step 4: Rank and filter leads using AI
            if leads:
                leads = self._rank_leads(leads, topic)
            
            st.write(f"✅ Found {len(leads)} potential leads")
            return leads[:num_leads]
            
        except Exception as e:
            st.error(f"Error in lead finding process: {str(e)}")
            return leads
    
    def _rank_leads(self, leads: List[Dict], topic: str) -> List[Dict]:
        """Rank leads by relevance using AI"""
        try:
            # Use OpenAI to score relevance
            leads_text = json.dumps([{k: v for k, v in lead.items() if k != 'relevance_score'} for lead in leads])
            
            prompt = f"""
            Rate the relevance of these leads for the topic "{topic}" on a scale of 0.0 to 1.0.
            Consider the person's expertise, role, and description relevance.
            
            Leads: {leads_text}
            
            Return only a JSON array of relevance scores in the same order, like: [0.9, 0.7, 0.85, ...]
            """
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            
            scores = json.loads(response.choices[0].message.content)
            
            # Assign scores to leads
            for i, score in enumerate(scores):
                if i < len(leads):
                    leads[i]['relevance_score'] = score
            
            # Sort by relevance score
            leads.sort(key=lambda x: x['relevance_score'], reverse=True)
            
        except Exception as e:
            st.warning(f"Could not rank leads with AI: {str(e)}")
            # Fall back to original order
        
        return leads

def main():
    """Streamlit interface for LeadFinder AI"""
    st.set_page_config(page_title="LeadFinder AI", page_icon="🎯", layout="wide")
    
    st.title("🎯 LeadFinder AI")
    st.markdown("*Automatically identify subject-matter experts using AI-powered search*")
    
    # Check for API keys
    if not EXA_API_KEY or not OPENAI_API_KEY:
        st.error("Please set your EXA_API_KEY and OPENAI_API_KEY in the .env file")
        st.stop()
    
    # Initialize agent
    if 'agent' not in st.session_state:
        st.session_state.agent = LeadFinderAgent()
    
    # Sidebar for settings
    st.sidebar.header("⚙️ Settings")
    
    # Main input
    topic = st.sidebar.text_input(
        "Topic/Expertise Area",
        placeholder="e.g., blockchain security, sustainable packaging",
        help="Enter the subject matter or expertise you're looking for"
    )
    
    num_leads = st.sidebar.slider("Number of leads", min_value=1, max_value=50, value=10)
    
    search_type = st.sidebar.selectbox(
        "Search Type",
        ["auto", "neural", "keyword", "fast"],
        help="Auto combines neural and keyword search for best results"
    )
    
    use_linkedin = st.sidebar.checkbox("Include LinkedIn search", value=True)
    use_full_text = st.sidebar.checkbox("Enable full-text analysis", value=True)
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("🔍 Find Leads", type="primary", disabled=not topic):
            if topic:
                with st.spinner("Searching for leads..."):
                    leads = st.session_state.agent.find_leads(
                        topic=topic,
                        num_leads=num_leads,
                        search_type=search_type,
                        use_linkedin=use_linkedin,
                        use_full_text=use_full_text
                    )
                    
                    st.session_state.leads = leads
            else:
                st.warning("Please enter a topic to search for.")
    
    with col2:
        if st.button("🔄 Clear Results"):
            if 'leads' in st.session_state:
                del st.session_state.leads
            st.rerun()
    
    # Display results
    if 'leads' in st.session_state and st.session_state.leads:
        st.header("📋 Found Leads")
        
        # Convert to DataFrame for display
        df = pd.DataFrame(st.session_state.leads)
        
        # Reorder columns for better display
        column_order = ['name', 'company_role', 'linkedin_url', 'phone', 'email', 
                       'description', 'relevance_score', 'source_url']
        df = df.reindex(columns=[col for col in column_order if col in df.columns])
        
        # Display table
        st.dataframe(
            df,
            width=True,
            column_config={
                "linkedin_url": st.column_config.LinkColumn("LinkedIn"),
                "source_url": st.column_config.LinkColumn("Source"),
                "relevance_score": st.column_config.NumberColumn("Relevance", format="%.2f"),
            }
        )
        
        # Download button
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name=f"leads_{topic.replace(' ', '_')}.csv",
            mime="text/csv"
        )
        
        # Summary stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Leads", len(df))
        with col2:
            linkedin_count = df['linkedin_url'].notna().sum()
            st.metric("With LinkedIn", linkedin_count)
        with col3:
            contact_count = (df['phone'].notna() | df['email'].notna()).sum()
            st.metric("With Contact Info", contact_count)
    
    elif 'leads' in st.session_state:
        st.info("No leads found for this topic. Try adjusting your search terms or settings.")
    
    # Help section
    with st.expander("ℹ️ How to use LeadFinder AI"):
        st.markdown("""
        ### Getting Started
        1. **Enter a topic** in the sidebar (e.g., "AI ethics", "fintech innovation")
        2. **Adjust settings** like number of leads and search type
        3. **Click "Find Leads"** to start the search process
        
        ### Search Types
        - **Auto**: Combines neural and keyword search (recommended)
        - **Neural**: Uses AI embeddings for semantic matching
        - **Keyword**: Traditional keyword-based search
        - **Fast**: Optimized for speed with good results
        
        ### Features
        - 🔍 **Web Search**: Uses Exa's AI-powered search engine
        - 👔 **LinkedIn Integration**: Finds professional profiles
        - 📞 **Contact Extraction**: Parses phone numbers and emails
        - 🤖 **AI Ranking**: Scores leads by relevance
        - 📊 **Export**: Download results as CSV
        
        ### Tips for Better Results
        - Use specific expertise areas (e.g., "blockchain security" vs "technology")
        - Try different search types if results aren't relevant
        - Enable full-text analysis for better contact extraction
        """)

if __name__ == "__main__":
    main()