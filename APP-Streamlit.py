import streamlit as st
import os
from RAG_News import categorize, get_linksDB, get_news, get_news_GEN, TextSummarizationPipeline

st.title("📰 News Summarization App")
st.write("Easily summarize news articles based on keywords or by providing a link.")

st.write("---")

st.subheader("Select an Option:")
option = st.selectbox(
    "How would you like to summarize news?",
    ("Search by Keywords 🔍", "Enter a Web Link 🌐")
)

st.write("")  

if option == "Search by Keywords 🔍":
    user_prompt = st.text_input("Enter your search keywords below:", placeholder="e.g., AI in healthcare")
    user_link = None  

elif option == "Enter a Web Link 🌐":
    user_link = st.text_input("Enter a web link below:", placeholder="e.g., https://example.com/news-article")
    user_prompt = None  

st.write("") 

# Summarize button
if st.button("🔎 Summarize News"):
    if option == "Search by Keywords 🔍" and user_prompt:
        model_name = 'meta/meta-llama-3-8b-instruct'
        category = categorize(user_prompt, model_name)
        st.write(f"🗂️ Your prompt is categorized under: {category if category else 'General'}")

        if category is None:
            category = "General"  

        allowed_categories = ['Technology', 'Science', 'Health', 'Sports']

        if category not in allowed_categories:
            links = get_linksDB(category, user_prompt)
            if not links:
                st.write("⚠️ No related news found.")
            else:
                summarizer = TextSummarizationPipeline()

                for link in links:
                    news_content = get_news_GEN(link, user_prompt)

                    summary = summarizer.generate_summary(news_content)
                    st.write(summary[0]['generated_text'])
                    st.write("---")
                    break
        else:
            # Retrieve news links
            links = get_linksDB(category, user_prompt)
            if not links:
                st.write("⚠️ No related news found.")
            else:
                summarizer = TextSummarizationPipeline()

                for link in links:
                    news_content = get_news(link)

                    summary = summarizer.generate_summary(news_content)
                    st.write(f"🔗 [Read Article]({link})")
                    st.write(summary[0]['generated_text'])
                    st.write("---")

    elif option == "Enter a Web Link 🌐" and user_link:
        news_content = get_news(user_link)
        if news_content:
            summarizer = TextSummarizationPipeline()
            summary = summarizer.generate_summary(news_content)
            st.write(f"🔗 [Read Article]({user_link})")
            st.write(summary[0]['generated_text'])
            st.write("---")
        else:
            st.write("⚠️ Failed to retrieve content from the link.")
    else:
        st.write("⚠️ Please enter valid keywords or a link.")
