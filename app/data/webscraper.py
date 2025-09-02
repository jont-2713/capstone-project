import argparse
import json
import sys
import time
from typing import Dict, List

try:
    import instaloader
    from instaloader import Profile
    from instaloader.exceptions import (
        TooManyRequestsException,
        ProfileNotExistsException,
        PrivateProfileNotFollowedException,
    )
except ImportError as exc:
    sys.stderr.write(
        "Error: The instaloader package is required to run this script.\n"
        "Install it with `pip install instaloader` and try again.\n"
    )
    raise


def login(
    loader: instaloader.Instaloader,
    username: str,
    password: str,
    sessionfile: str | None = None,
) -> None:
    """Log in with the given credentials or load an existing session.

    Instaloader will store the resulting session cookies in a cache file
    so that subsequent runs reuse the session instead of re‑authenticating.

    :param loader: Instaloader instance used for scraping.
    :param username: Instagram login username.
    :param password: Instagram login password.
    """
    # If the caller provides an explicit sessionfile, try to load it first.
    # This allows reuse of cookies exported from a real browser (e.g. via
    # Instaloader's 615_import_firefox_session.py helper).  Falling back to
    # Instaloader's built‑in login when no sessionfile is available.
    loaded = False
    try:
        if sessionfile:
            loader.load_session_from_file(username, sessionfile)
            loaded = True
        else:
            loader.load_session_from_file(username)
            loaded = True
        # Verify session is valid by checking that a username is bound to
        # the context.  Without a valid session, loader.context.username will be
        # ``None``.
        if loader.context.username is None:
            raise instaloader.exceptions.LoginException("session invalid")
    except Exception:
        loaded = False
    if loaded:
        loader.context.log(
            f"Reusing cached session for {username}" + (f" from {sessionfile}" if sessionfile else "") + "."
        )
        return
    # If session loading failed, fall back to password login.
    loader.context.log(f"Logging in as {username}…")
    loader.login(username, password)
    # Save session for reuse in default location or specified sessionfile
    loader.save_session_to_file(sessionfile or username)


def scrape_profile(
    loader: instaloader.Instaloader,
    profile: Profile,
    posts_limit: int = 5,
    sleep_between_posts: float = 5.0,
    rate_limit_sleep: float = 900.0,
) -> Dict[str, object]:
    """Collect metadata and recent posts for a given profile.

    :param loader: Instaloader instance.
    :param profile: Profile object to scrape.
    :param posts_limit: Maximum number of recent posts to retrieve.
    :param sleep_between_posts: Seconds to wait between successive post requests.
    :param rate_limit_sleep: Seconds to wait when a TooManyRequestsException occurs.
    :returns: Dictionary of scraped data.
    """
    data: Dict[str, object] = {
        "username": profile.username,
        "full_name": profile.full_name,
        "biography": profile.biography,
        "followers": profile.followers,
        "followees": profile.followees,
        "total_posts": profile.mediacount,
        "external_url": profile.external_url,
    }
    posts_data: List[Dict[str, object]] = []
    count = 0
    posts = profile.get_posts()
    while count < posts_limit:
        try:
            post = next(posts)
        except StopIteration:
            break
        try:
            caption = post.caption or ""
            image_url = post.url  # URL to the main image (for videos this is the thumbnail)
            posts_data.append(
                {
                    "shortcode": post.shortcode,
                    "caption": caption,
                    "image_url": image_url,
                    "likes": post.likes,
                    "date_utc": post.date_utc.isoformat(),
                }
            )
            count += 1
            time.sleep(sleep_between_posts)
        except TooManyRequestsException:
            loader.context.error("Rate limit reached while fetching posts. Sleeping…")
            time.sleep(rate_limit_sleep)
            continue
    data["recent_posts"] = posts_data
    return data


def run_scraper(config: Dict[str, object]) -> bool:
    """Run the Instagram scraper with the provided configuration.
    
    :param config: Dictionary containing scraper configuration:
        - target_username: Username to scrape
        - login_username: Instagram login username
        - login_password: Instagram login password
        - posts_limit: Number of posts to fetch (default: 5)
        - output_file: Output JSON file path
        - sessionfile: Optional session file path
    :returns: True if successful, False otherwise
    """
    try:
        # Extract configuration
        target_username = config['target_username']
        login_username = config['login_username']
        login_password = config['login_password']
        posts_limit = config.get('posts_limit', 5)
        output_file = config.get('output_file', 'instagram_data.json')
        sessionfile = config.get('sessionfile')
        
        # Initialize loader and login
        loader = instaloader.Instaloader()
        login(loader, login_username, login_password, sessionfile=sessionfile)
        
        # Scrape the target profile
        print(f"Scraping {target_username}…")
        profile = Profile.from_username(loader.context, target_username)
        result = scrape_profile(loader, profile, posts_limit=posts_limit)
        
        # Save results
        results = {target_username: result}
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
            
        print(f"Scraping complete. Data saved to {output_file}.")
        return True
        
    except ProfileNotExistsException:
        print(f"Profile '{target_username}' does not exist.")
        return False
    except PrivateProfileNotFollowedException:
        print(f"Profile '{target_username}' is private or not followed by the login user.")
        return False
    except TooManyRequestsException:
        # If a rate limit is hit at the profile level, wait and retry once
        print("Rate limit reached while accessing profile. Sleeping and retrying…")
        time.sleep(600)
        try:
            profile = Profile.from_username(loader.context, target_username)
            result = scrape_profile(loader, profile, posts_limit=posts_limit)
            results = {target_username: result}
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"Scraping complete after retry. Data saved to {output_file}.")
            return True
        except Exception as e:
            print(f"Failed to scrape {target_username} after retry: {e}")
            return False
    except Exception as e:
        print(f"Unexpected error while scraping {target_username}: {e}")
        return False


def get_target_profile_interactively() -> str:
    """Interactively get a single target profile from user input.
    
    :returns: Target username to scrape.
    """
    print("\n=== Instagram Profile Scraper ===")
    
    while True:
        username = input("Enter username to scrape: ").strip()
        if username:
            return username
        print("Please enter a valid username.")


def main(argv: List[str] | None = None) -> None:
    """Parse arguments and orchestrate scraping based on user input."""
    parser = argparse.ArgumentParser(description="Scrape Instagram profiles using Instaloader.")
    parser.add_argument(
        "--username",
        required=True,
        help="Instagram login username (the account used to authenticate).",
    )
    parser.add_argument(
        "--password",
        required=True,
        help="Instagram login password.",
    )
    parser.add_argument(
        "--sessionfile",
        help=(
            "Path to an Instaloader session file to reuse. If provided, the script "
            "will skip the login step and load cookies from this file. Use the "
            "615_import_firefox_session.py helper from Instaloader's documentation to "
            "create a session file from your browser."
        ),
    )
    parser.add_argument(
        "--posts",
        type=int,
        default=5,
        help="Number of recent posts to fetch per profile (default: 5).",
    )
    parser.add_argument(
        "--output",
        default="instagram_data.json",
        help="Path to output JSON file (default: instagram_data.json).",
    )
    args = parser.parse_args(argv)

    loader = instaloader.Instaloader()
    login(loader, args.username, args.password, sessionfile=args.sessionfile)

    # Get target profile interactively
    target = get_target_profile_interactively()

    print(f"\nProfile to scrape: {target}")
    confirm = input("Proceed with scraping? (y/n): ").strip().lower()
    if confirm not in ['y', 'yes']:
        print("Scraping cancelled.")
        return

    results: Dict[str, object] = {}
    try:
        profile = Profile.from_username(loader.context, target)
        print(f"Scraping {target}…")
        results[target] = scrape_profile(loader, profile, posts_limit=args.posts)
    except ProfileNotExistsException:
        print(f"Profile '{target}' does not exist.")
        return
    except PrivateProfileNotFollowedException:
        print(f"Profile '{target}' is private or not followed by the login user.")
        return
    except TooManyRequestsException:
        # If a rate limit is hit at the profile level, wait and retry once
        print("Rate limit reached while accessing profile. Sleeping and retrying…")
        time.sleep(600)
        try:
            profile = Profile.from_username(loader.context, target)
            results[target] = scrape_profile(loader, profile, posts_limit=args.posts)
        except Exception as e:
            print(f"Failed to scrape {target} after retry: {e}")
            return
    except Exception as e:
        print(f"Unexpected error while scraping {target}: {e}")
        return

    # Write collected data to JSON file
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Scraping complete. Data saved to {args.output}.")