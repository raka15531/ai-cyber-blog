# main.py
import os
import json
import feedparser
from datetime import datetime
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate

# Setup LLM
llm = Ollama(model="llama3", base_url="http://localhost:11434")
embeddings = OllamaEmbeddings(model="llama3", base_url="http://localhost:11434")

# Vector DB for memory
vector_db = Chroma(persist_directory="./vector_db", embedding_function=embeddings)

# Track published topics
PUBLISHED_FILE = "published.json"
if os.path.exists(PUBLISHED_FILE):
    with open(PUBLISHED_FILE, "r") as f:
        PUBLISHED_TOPICS = json.load(f)
else:
    PUBLISHED_TOPICS = []

# Folder to save drafts
DRAFTS_DIR = "drafts"
os.makedirs(DRAFTS_DIR, exist_ok=True)

# === Fetch News ===
def fetch_trending_news():
    feeds = [
        "https://feeds.feedburner.com/TheHackersNews",
        "https://krebsonsecurity.com/feed/"
    ]
    articles = []
    for url in feeds:
        feed = feedparser.parse(url)
        for entry in feed.entries[:3]:
            if entry.title not in PUBLISHED_TOPICS:
                articles.append({
                    "title": entry.title,
                    "summary": entry.summary,
                    "link": entry.link
                })
    return articles

# === Generate Blog with Memory ===
def generate_blog_with_context(news_item):
    results = vector_db.similarity_search(news_item["title"], k=1)
    context = results[0].page_content if results else "No prior content."

    prompt = PromptTemplate.from_template("""
    Write a 700-word blog post about this cybersecurity news. Use past content as style guide.

    News: {title}
    Summary: {summary}

    Previous post snippet: {context}

    Structure:
    - Engaging intro
    - Technical details
    - Real-world impact
    - How to defend
    - Closing thoughts

    Use HTML-like headings (H2/H3) and bullet points.
    """)
    chain = prompt | llm
    return chain.invoke({
        "title": news_item["title"],
        "summary": news_item["summary"],
        "context": context
    })

# === Save Draft to File ===
def save_draft(title, content):
    filename = f"{DRAFTS_DIR}/{title[:50].replace(' ', '_').replace(':', '')}.html"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"<!-- Generated on {datetime.now().isoformat()} -->\n")
        f.write(content)
    print(f"💾 Draft saved: {filename}")
    return filename

# === Save to Memory ===
def save_to_vector_db(title, content):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_text(content)
    vector_db.add_texts(chunks, metadatas=[{"title": title}] * len(chunks))
    vector_db.persist()

# === Main Workflow ===
def run_automation():
    print(f"[{datetime.now()}] 🌐 Fetching latest cybersecurity news...")
    news_items = fetch_trending_news()

    if not news_items:
        print("📭 No new articles to process.")
        return

    news_item = news_items[0]
    print(f"✍️ Writing draft: {news_item['title']}")

    try:
        content = generate_blog_with_context(news_item)
        filename = save_draft(news_item["title"], content)
        save_to_vector_db(news_item["title"], content)

        PUBLISHED_TOPICS.append(news_item["title"])
        with open(PUBLISHED_FILE, "w") as f:
            json.dump(PUBLISHED_TOPICS, f)

        print(f"✅ Draft created and memory updated.")

    except Exception as e:
        print(f"💥 Error: {e}")

if __name__ == "__main__":
    run_automation()
