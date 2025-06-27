import joblib
import pandas as pd
from datetime import datetime

# Load trained model and vectorizer
model = joblib.load("mentor_classifier.pkl")
vectorizer = joblib.load("mentor_vectorizer.pkl")

# Load IT blogs CSV
it_blogs = pd.read_csv("it_blogs.csv")

def get_chatbot_response(message):
    # Vectorize user message
    X = vectorizer.transform([message])
    category = model.predict(X)[0]

    # Build response based on predicted category
    if category == "linkedin_growth":
        posts = it_blogs.sample(2).iloc[:, 0].tolist()
        hashtags = "#CareerTips #LinkedInGrowth #Networking"
        return f"📌 LinkedIn Growth Tips:\n- {posts[0]}\n- {posts[1]}\nUse hashtags like {hashtags}."

    elif category == "github_trending":
        return "🚀 Congrats! You're trending on GitHub. Share your project story, thank contributors and use hashtags like #GitHubTrending #OpenSource."

    elif category == "linkedin_content_tip":
        return "📈 Pro Tip: Use personal stories, polls, and calls-to-action in your LinkedIn posts for better engagement."

    elif category == "github_activity":
        return "👨‍💻 Keep contributing to open-source repos. You can post updates about your issues fixed and repos starred!"

    elif category == "greeting":
        return "👋 Hello! How can I assist you today?"

    elif category == "farewell":
        return "👋 See you soon. Have a great day!"

    elif category == "small_talk":
        return "🙂 I'm just a bot, but I'm doing great! How about you?"

    elif category == "time_query":
        return f"🕒 Current time is {datetime.now().strftime('%H:%M:%S')}."

    elif category == "weather_query":
        return "☁️ I can't fetch real-time weather yet, but it’s always a good idea to check a weather app!"

    else:
        return "🔥 Keep learning and sharing! What else can I help you with?"

# Test example
print(get_chatbot_response("How do I grow my LinkedIn connections fast?"))
