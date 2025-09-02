import sys
import os
from app.data.webscraper import run_scraper

def main():
    """Main pipeline that collects username and runs the Instagram scraper."""
    print("=== Social Media Analysis Pipeline ===")
    
    # Get username input
    while True:
        username = input("Enter a profile username: ").strip()
        if username:
            break
        print("Please enter a valid username.")
    
    print(f"Starting analysis for: {username}")
    
    # Configure scraper settings (you can modify these as needed)
    scraper_config = {
        'target_username': username,
        'login_username': "mewodal352",  # Replace with your Instagram login
        'login_password': "Ronnie12",  # Replace with your Instagram password
        'posts_limit': 5,
        'output_file': f'scraped_data.json',
        'sessionfile': "my_instagram_session"  # Optional: path to session file
    }
    
    try:
        # Run the Instagram scraper
        print("Running Instagram scraper...")
        success = run_scraper(scraper_config)
        
        if success:
            print(f"? Successfully scraped profile: {username}")
            print(f"Data saved to: {scraper_config['output_file']}")
            
            # Here you can add additional pipeline steps, such as:
            # - Data analysis
            # - Report generation
            # - Database storage
            # etc.
            
        else:
            print(f"? Failed to scrape profile: {username}")
            
    except Exception as e:
        print(f"Error in pipeline: {e}")
        return False
    
    return True

if __name__ == "__main__":
    main()
