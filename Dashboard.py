import streamlit as st
import instaloader
import time

# Initialize Instaloader
L = instaloader.Instaloader()

L.context._session.cookies.set("sessionid", "77091777356%3AGvYRV8iFJcEqAa%3A16%3AAYfD62iYCYgWu0Xm0LssfSDXggx5akINZ6_6OL_Jbw")

st.title("Instagram Scraper with Instaloader")

# Input field for username(s)
usernames = st.text_input("Enter Instagram usernames (comma-separated):")
max_posts = st.slider("Max posts per user", 5, 100, 20)

if st.button("Download Posts"):
    if usernames:
        usernames = [u.strip() for u in usernames.split(",")]
        for username in usernames:
            try:
                profile = instaloader.Profile.from_username(L.context, username)
                st.write(f"**User:** {profile.username}")
                st.write(f"Followers: {profile.followers}")
                st.write(f"Following: {profile.followees}")
                st.write(f"Posts: {profile.mediacount}")

                # Show first few posts
                posts = profile.get_posts()
                for i, post in enumerate(posts, start=1):
                    if i > 3:  # show only first 3 posts
                        break
                    st.image(post.url, caption=post.caption[:100] + "...")
            except Exception as e:
                st.error(f"Failed for {username}: {e}")
    else:
        st.warning("Please enter at least one username.")