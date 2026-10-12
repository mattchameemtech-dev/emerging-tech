import json
import os
import sys
from datetime import datetime
import feedparser
from openai import OpenAI

# ----------------------------------------------------
# CONFIGURATION: Targeted Military R&D Data Sources
# ----------------------------------------------------
RSS_FEEDS = {
    "DARPA News": "https://darpa.mil",
    "DoD Press Releases": "https://defense.gov",
    "DIU Activity": "https://diu.mil"
}

def gather_raw_military_data():
    """Fetches the latest articles from defense technology RSS feeds."""
    print("Gathering raw military research feeds...")
    collected_articles = []
    
    for source_name, url in RSS_FEEDS.items():
        try:
            feed = feedparser.parse(url)
            # Take the top 4 most recent articles from each source
            for entry in feed.entries[:4]:
                title = entry.get("title", "")
                summary = entry.get("summary", "") or entry.get("description", "")
                link = entry.get("link", "")
                
                collected_articles.append({
                    "source": source_name,
                    "title": title,
                    "summary": summary,
                    "link": link
                })
        except Exception as e:
            print(f"Warning: Could not pull from {source_name}: {e}", file=sys.stderr)
            
    return collected_articles

def analyze_with_gemini(raw_data):
    """Sends raw text to Gemini 2.5 Flash to extract and synthesize trends."""
    api_key = os.environ.get("AI_API_KEY")
    if not api_key:
        print("Error: AI_API_KEY environment variable is missing!", file=sys.stderr)
        sys.exit(1)
        
    client = OpenAI(
        api_key=api_key,
        base_url="https://googleapis.com"
    )
    
    # Structure the payload as a string for the prompt context
    serialized_articles = json.dumps(raw_data, indent=2)
    
    system_instruction = (
        "You are an elite US Government Military R&D Analyst. Your job is to read raw defense press releases "
        "and find the underlying technology breakthroughs, trends, or major project milestones. "
        "You must respond ONLY with a valid JSON array matching this exact format: "
        '[{"title": "Trend Name", "summary": "Detailed technical analysis...", "source": "Source Name (e.g. DARPA)"}]'
    )
    
    user_prompt = f"Analyze the following recent raw data and output the top 3-5 macro R&D trends:\n\n{serialized_articles}"
    
    print("Consulting Gemini 2.5 Flash for strategic synthesis...")
    try:
        response = client.chat.completions.create(
            model="gemini-2.5-flash",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2, # Low temperature ensures structured data consistency
            response_format={"type": "json_object"}
        )
        
        raw_output = response.choices[0].message.content
        # Clean up any potential markdown backticks that the model might wrap around JSON
        if raw_output.startswith("```json"):
            raw_output = raw_output.replace("```json", "").replace("```", "").strip()
            
        return json.loads(raw_output)
    except Exception as e:
        print(f"Error during AI analysis: {e}", file=sys.stderr)
        sys.exit(1)

def main():
    # Phase 1: Collect
    raw_articles = gather_raw_military_data()
    if not raw_articles:
        print("No articles collected. Exiting execution.")
        return

    # Phase 2: Synthesize
    ai_trends = analyze_with_gemini(raw_articles)
    
    # If Gemini outputs a dictionary wrap under a key, handle it gracefully
    if isinstance(ai_trends, dict) and "trends" in ai_trends:
        trends_list = ai_trends["trends"]
    elif isinstance(ai_trends, dict):
        trends_list = list(ai_trends.values())[0] if isinstance(list(ai_trends.values())[0], list) else [ai_trends]
    else:
        trends_list = ai_trends

    # Phase 3: Deliver static artifact
    payload = {
        "last_updated": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        "trends": trends_list
    }
    
    with open("data.json", "w") as f:
        json.dump(payload, f, indent=2)
        
    print(f"Success! Saved {len(trends_list)} fresh trends to data.json.")

if __name__ == "__main__":
    main()
