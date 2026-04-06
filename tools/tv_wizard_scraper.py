"""
Pine Wizards Bulk Scraper
=========================
Scrapes all Pine Script Wizard profiles from https://www.tradingview.com/pine-wizards/
then downloads their most popular open-source indicators and strategies.

Usage:
    python tools/tv_wizard_scraper.py                    # Scrape all wizards
    python tools/tv_wizard_scraper.py --min-boosts 200   # Lower boost threshold
    python tools/tv_wizard_scraper.py --strategies        # Include strategies (default: indicators only)
    python tools/tv_wizard_scraper.py --authors LuxAlgo,LazyBear  # Only specific authors
    python tools/tv_wizard_scraper.py --max-per-author 10          # Limit scripts per author
    python tools/tv_wizard_scraper.py --dry-run            # List what would be downloaded
"""

import asyncio
import argparse
import json
import re
import os
import sys
from pathlib import Path
from datetime import datetime
from playwright.async_api import async_playwright

# ── Defaults ──────────────────────────────────────────────────────────────────
WIZARDS_URL = "https://www.tradingview.com/pine-wizards/"
DEFAULT_MIN_BOOSTS = 100
DEFAULT_MAX_PER_AUTHOR = 50  # 0 = unlimited
BASE_OUTPUT_DIR = Path("indicators")
MANIFEST_FILE = BASE_OUTPUT_DIR / "wizard_manifest.json"

# Static analysis filters — reject scripts with these patterns
FORBIDDEN_PATTERNS = [
    r"request\.security",
    r"security\(",
    r"lookahead\s*=\s*barmerge\.lookahead_on",
    r"calc_on_every_tick\s*=\s*true",
    r"calc_on_order_fills\s*=\s*true",
    r"varip\s+",
]


async def extract_wizard_profiles(page) -> list[dict]:
    """Extract all wizard profile links from the Pine Wizards page."""
    print(f"Navigating to {WIZARDS_URL}...")
    await page.goto(WIZARDS_URL, wait_until="networkidle")
    await asyncio.sleep(3)

    # Scroll to load full page
    await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
    await asyncio.sleep(2)

    # Extract all profile links — they follow the pattern /u/USERNAME/
    wizards = []
    seen = set()

    # Method 1: Look for links to /u/USERNAME/ profiles
    links = await page.query_selector_all('a[href*="/u/"]')
    for link in links:
        href = await link.get_attribute("href") or ""
        match = re.search(r"/u/([^/#?]+)", href)
        if match:
            username = match.group(1)
            if username not in seen:
                seen.add(username)
                # Try to get display name from link text
                text = (await link.inner_text()).strip()
                wizards.append({
                    "username": username,
                    "display_name": text if text and text != username else username,
                    "profile_url": f"https://www.tradingview.com/u/{username}/#published-scripts",
                })

    print(f"Found {len(wizards)} Pine Wizards: {', '.join(w['username'] for w in wizards)}")
    return wizards


async def scrape_author_scripts(page, author: dict, args) -> list[dict]:
    """Scrape all qualifying scripts from a single author's profile."""
    username = author["username"]
    url = author["profile_url"]
    output_dir = BASE_OUTPUT_DIR / username

    print(f"\n{'='*60}")
    print(f"Scraping: {username}")
    print(f"{'='*60}")

    await page.goto(url, wait_until="networkidle")
    await asyncio.sleep(2)

    # Scroll to load all script cards
    print("  Scrolling to load all scripts...")
    scroll_attempts = 0
    while scroll_attempts < 50:  # Safety limit
        last_count = len(await page.query_selector_all('article[class*="card-"]'))
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        await asyncio.sleep(3)
        new_count = len(await page.query_selector_all('article[class*="card-"]'))
        if new_count > last_count:
            print(f"    Loaded {new_count} scripts...")
            scroll_attempts = 0  # Reset on progress
        else:
            scroll_attempts += 1
            if scroll_attempts >= 2:
                break

    # Extract script cards
    cards = await page.query_selector_all('article[class*="card-"]')
    print(f"  Found {len(cards)} total scripts. Filtering (min_boosts={args.min_boosts})...")

    scripts_to_fetch = []
    for card in cards:
        title_el = await card.query_selector('a[data-qa-id="ui-lib-card-link-title"]')
        if not title_el:
            continue

        title = await title_el.inner_text()
        href = await title_el.get_attribute("href")
        link = href if href.startswith("http") else f"https://www.tradingview.com{href}"

        # Boost count
        boost_count = 0
        boost_label_el = await card.query_selector('[aria-label*="boosts"]')
        if boost_label_el:
            aria_label = await boost_label_el.get_attribute("aria-label") or ""
            m = re.search(r"([\d,]+)", aria_label)
            if m:
                boost_count = int(m.group(1).replace(",", ""))
        else:
            boost_btn = await card.query_selector('[data-qa-id="ui-lib-card-like-button"]')
            if boost_btn:
                aria_label = await boost_btn.get_attribute("aria-label") or ""
                m = re.search(r"([\d,]+)", aria_label)
                if m:
                    boost_count = int(m.group(1).replace(",", ""))

        # Category (indicator vs strategy)
        category_el = await card.query_selector('span[class*="visuallyHiddenLabel-"]')
        category = await category_el.inner_text() if category_el else ""
        is_strategy = "strategy" in category.lower()

        # Filter: boost threshold
        if boost_count < args.min_boosts:
            continue

        # Filter: indicators only (unless --strategies flag)
        if not args.strategies and is_strategy:
            continue

        scripts_to_fetch.append({
            "title": title,
            "link": link,
            "boosts": boost_count,
            "is_strategy": is_strategy,
            "author": username,
        })

    # Sort by boosts descending
    scripts_to_fetch.sort(key=lambda x: x["boosts"], reverse=True)

    # Apply max-per-author limit
    if args.max_per_author > 0:
        scripts_to_fetch = scripts_to_fetch[:args.max_per_author]

    print(f"  {len(scripts_to_fetch)} scripts pass filters.")

    if args.dry_run:
        for s in scripts_to_fetch:
            tag = "STRAT" if s["is_strategy"] else "IND"
            print(f"    [{tag}] {s['title']} ({s['boosts']} boosts)")
        return scripts_to_fetch

    # Download source code for each script
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    downloaded = []
    for i, script in enumerate(scripts_to_fetch):
        tag = "STRAT" if script["is_strategy"] else "IND"
        print(f"  [{i+1}/{len(scripts_to_fetch)}] [{tag}] {script['title']} ({script['boosts']} boosts)...")
        try:
            await page.goto(script["link"], wait_until="networkidle")
            await asyncio.sleep(3)

            # Click 'Source code' tab
            code_tab = await page.query_selector('#code')
            if code_tab:
                await code_tab.click()
                await asyncio.sleep(2)

            # Extract source code
            source_box = (
                await page.query_selector('div[class*="code-G0eoc001"]')
                or await page.query_selector('div[class*="code-"]')
                or await page.query_selector('.tv-community-scripts__source-code-inner')
            )

            if not source_box:
                print(f"    [Error] Could not find source code — may be closed-source")
                continue

            source_code = await source_box.inner_text()

            # Static analysis filter
            rejected = False
            for pattern in FORBIDDEN_PATTERNS:
                if re.search(pattern, source_code, re.IGNORECASE):
                    print(f"    [Skipped] Forbidden pattern: {pattern}")
                    rejected = True
                    break
            if rejected:
                continue

            # Save
            safe_name = re.sub(r'[^\w\-_\. ]', '_', script["title"]).replace(' ', '_')
            filename = output_dir / f"{safe_name}.pine"

            with open(filename, "w", encoding="utf-8") as f:
                f.write(f"// Title: {script['title']}\n")
                f.write(f"// Author: {username}\n")
                f.write(f"// Boosts: {script['boosts']}\n")
                f.write(f"// Type: {'Strategy' if script['is_strategy'] else 'Indicator'}\n")
                f.write(f"// Link: {script['link']}\n")
                f.write(f"// Scraped: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
                f.write(source_code)

            print(f"    [Saved] -> {filename}")
            script["filename"] = str(filename)
            downloaded.append(script)

        except Exception as e:
            print(f"    [Error] {e}")

    return downloaded


async def main():
    parser = argparse.ArgumentParser(description="Pine Wizards Bulk Scraper")
    parser.add_argument("--min-boosts", type=int, default=DEFAULT_MIN_BOOSTS,
                        help=f"Minimum boosts/likes required (default: {DEFAULT_MIN_BOOSTS})")
    parser.add_argument("--max-per-author", type=int, default=DEFAULT_MAX_PER_AUTHOR,
                        help=f"Max scripts per author, 0=unlimited (default: {DEFAULT_MAX_PER_AUTHOR})")
    parser.add_argument("--strategies", action="store_true",
                        help="Include strategies (default: indicators only)")
    parser.add_argument("--authors", type=str, default="",
                        help="Comma-separated list of specific authors to scrape (default: all wizards)")
    parser.add_argument("--dry-run", action="store_true",
                        help="List matching scripts without downloading")
    parser.add_argument("--headless", action="store_true",
                        help="Run browser in headless mode")
    args = parser.parse_args()

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=args.headless)
        page = await browser.new_page()

        # Step 1: Get wizard profiles
        wizards = await extract_wizard_profiles(page)

        # Filter to specific authors if requested
        if args.authors:
            author_filter = {a.strip().lower() for a in args.authors.split(",")}
            wizards = [w for w in wizards if w["username"].lower() in author_filter]
            if not wizards:
                print(f"No matching wizards found for: {args.authors}")
                await browser.close()
                return

        # Step 2: Scrape each wizard's scripts
        manifest = {
            "scraped_at": datetime.now().isoformat(),
            "min_boosts": args.min_boosts,
            "max_per_author": args.max_per_author,
            "include_strategies": args.strategies,
            "authors": {},
        }

        total_downloaded = 0
        for wizard in wizards:
            downloaded = await scrape_author_scripts(page, wizard, args)
            manifest["authors"][wizard["username"]] = {
                "display_name": wizard["display_name"],
                "scripts_found": len(downloaded),
                "scripts": downloaded,
            }
            total_downloaded += len(downloaded)

        await browser.close()

        # Save manifest
        if not args.dry_run:
            BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            with open(MANIFEST_FILE, "w") as f:
                json.dump(manifest, f, indent=2)
            print(f"\nManifest saved to {MANIFEST_FILE}")

        print(f"\n{'='*60}")
        print(f"COMPLETE: {total_downloaded} scripts from {len(wizards)} wizards")
        print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
