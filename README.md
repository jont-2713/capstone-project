Instagator: The Social Media Detective
AI-Powered Social Media Analysis for Digital Investigators

 1. Overview
Instagator is an investigative tool that helps analysts automatically scrape, analyze, and visualize social-media data.
 It combines AI-based text and image sentiment analysis with a custom risk-scoring system, allowing investigators to rapidly identify high-risk or suspicious content. 
Please note: that the newest version of python is required for this program to function properly.

 2. Getting Started
Step 1. Install dependencies
Before launching the program for the first time:
Double-click setup.bat
 -> This installs all required packages from requirements.txt.


This step only needs to be done once (or when new dependencies are added).


Step 2. Launch the application
Double-click run_app.bat.


The Streamlit web interface will open automatically in your default browser.
 Default URL: http://127.0.0.1:8501
Alternatively - in your python terminal run the common - “streamlit run .\main.py”




 3. Landing Page
The landing page welcomes investigators to Instagator and provides quick navigation:
Scraper – collect data from public Instagram profiles.


Analytics – explore results, view sentiment breakdowns, and assess risk scores.


Post Browser – manually review each scraped post with AI-generated insights.


You’ll also see the Instagator logo and “Quick Actions” buttons for instant navigation.

 4. Using the Scraper
Go to Scraper in the sidebar.


Enter one or more Instagram usernames (comma,separated).


Choose how many recent posts to analyse with the “Max posts per user” slider.


Click Download / Update Posts.


The app will:
Fetch the user’s profile, metadata, and posts.


Analyze each post’s text sentiment and image sentiment.


Calculate a combined Risk Score (0–100).


Cache the results locally for later review.


Note: Only public Instagram profiles can be scraped.

5. The Analytics Dashboard
Navigate to Analytics from the sidebar.
 Here you can:
View every scraped user’s profile summary (followers, posts, average risk).


Review risk-level distributions (Low / Medium / High).


Filter posts by sentiment (positive / neutral / negative) and by keywords.


Display posts as evenly sized visual tiles, each showing:


Image or video thumbnail


Caption sentiment


Image sentiment


Risk Score with color codes:


🟥 High = Error style


🟨 Medium = Warning style


🟩 Low = Success style



6. Post Browser
For deeper review:
Open Post Browser.


Browse all scraped posts individually.


Each post displays:


Image or video preview


Caption and hashtags


AI-generated text and image sentiment


Calculated risk band (Low / Medium / High)


Use this page for detailed case-by-case analysis or to export findings.

7. Understanding the Risk Score
The risk engine blends both text and image sentiment:
Factor
Effect on Score
Range
Negative text
+ 30 × sentiment confidence
Raises risk
Positive text
− 20 × sentiment confidence
Lowers risk
Negative image
+ 25 × confidence
Raises risk
Positive image
− 15 × confidence
Lowers risk
Neutral image
+ 5 × confidence
Minor bump

The final value (0–100) is then classified:
0 – 33 = Low


34 – 66 = Medium


67 – 100 = High



8. Clearing Cached Data
If you need to reset the workspace:
Click “Clear cached results” on the Scraper page.


All temporary data stored under /Static will be removed automatically when the app closes.



9. Tips for Investigators
Start with smaller post limits (10 – 20) before large-scale scrapes.


Use the keyword filter in Analytics to isolate relevant cases or terms (e.g., “violence”, “weapon”).


Compare caption vs image sentiment to detect inconsistencies (e.g., happy captions with negative imagery).


High risk clusters may indicate subjects of investigative interest.



10. Troubleshooting
Issue
Fix
ModuleNotFoundError
Re-run setup_env.bat
“Session expired” 
Update your sessionid cookie in 1_Scraper.py - see 10.1 for information
App won’t open
Run python –m streamlit run main.py manually
Missing images
Ensure Static/ folder isn’t deleted mid-run



10.1 Renewing sessionid
Go to instagram in your browser and log in

When on instagram press the F12 key - then navigate to application - cookies and copy the “sessionid” value to your clipboard


Then in a text editor open the “1_scraper.py” file in the “pages” folder press ctrl + f and search for “sessionid” paste the value in place of the existing one and ctrl + s to save

11. Closing the App
Simply close the Streamlit browser tab and the console window.


Temporary files are auto-cleaned on exit.

