"""
Quick discovery script: opens TradingView, you log in,
then it dumps all interactive elements (buttons, inputs, menus)
so we can find the correct selectors for the chart downloader.
"""

import asyncio
import threading
import tkinter as tk
from tkinter import messagebox
from playwright.async_api import async_playwright

def gui_prompt(message: str, title: str = "TV Discovery"):
    def _show():
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        messagebox.showinfo(title, message, parent=root)
        root.destroy()
    t = threading.Thread(target=_show, daemon=True)
    t.start()
    t.join()


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context()
        page = await context.new_page()
        await page.set_viewport_size({"width": 1920, "height": 1080})

        await page.goto("https://www.tradingview.com/chart/", wait_until="domcontentloaded")
        await asyncio.sleep(3)

        gui_prompt("Log in to TradingView.\n\nOnce you see the chart, click OK.")
        await asyncio.sleep(2)

        # ── Dump ALL buttons ──
        print("\n" + "=" * 80)
        print("ALL BUTTONS ON PAGE")
        print("=" * 80)
        buttons = await page.evaluate("""() => {
            return Array.from(document.querySelectorAll('button')).map((b, i) => ({
                index: i,
                text: b.textContent?.trim().substring(0, 80),
                ariaLabel: b.getAttribute('aria-label'),
                dataName: b.getAttribute('data-name'),
                dataTooltip: b.getAttribute('data-tooltip'),
                id: b.id,
                visible: b.offsetParent !== null,
                rect: b.getBoundingClientRect().toJSON(),
            }));
        }""")
        for b in buttons:
            if not b.get("visible"):
                continue
            parts = []
            if b.get("id"): parts.append(f"id='{b['id']}'")
            if b.get("dataName"): parts.append(f"data-name='{b['dataName']}'")
            if b.get("ariaLabel"): parts.append(f"aria-label='{b['ariaLabel']}'")
            if b.get("dataTooltip"): parts.append(f"tooltip='{b['dataTooltip']}'")
            text = b.get("text", "")
            if text:
                parts.append(f"text='{text[:40]}'")
            y = int(b.get("rect", {}).get("y", 0))
            if parts:
                print(f"  [{b['index']:3d}] y={y:4d} | {' | '.join(parts)}")

        # ── Dump toolbar area specifically ──
        print("\n" + "=" * 80)
        print("TOOLBAR ELEMENTS (top 100px)")
        print("=" * 80)
        toolbar = await page.evaluate("""() => {
            return Array.from(document.querySelectorAll('button, [role="button"], input, select'))
                .filter(el => el.getBoundingClientRect().y < 100)
                .map(el => ({
                    tag: el.tagName,
                    text: el.textContent?.trim().substring(0, 60),
                    ariaLabel: el.getAttribute('aria-label'),
                    dataName: el.getAttribute('data-name'),
                    dataTooltip: el.getAttribute('data-tooltip'),
                    id: el.id,
                    type: el.getAttribute('type'),
                    placeholder: el.getAttribute('placeholder'),
                    x: Math.round(el.getBoundingClientRect().x),
                    y: Math.round(el.getBoundingClientRect().y),
                    w: Math.round(el.getBoundingClientRect().width),
                }));
        }""")
        for el in toolbar:
            parts = [el['tag']]
            if el.get("id"): parts.append(f"id='{el['id']}'")
            if el.get("dataName"): parts.append(f"data-name='{el['dataName']}'")
            if el.get("ariaLabel"): parts.append(f"aria='{el['ariaLabel']}'")
            if el.get("dataTooltip"): parts.append(f"tooltip='{el['dataTooltip']}'")
            if el.get("placeholder"): parts.append(f"placeholder='{el['placeholder']}'")
            if el.get("text"): parts.append(f"text='{el['text'][:40]}'")
            print(f"  x={el['x']:4d} w={el['w']:3d} | {' | '.join(parts)}")

        # ── Now open the chart type dropdown and dump its contents ──
        print("\n" + "=" * 80)
        print("CHART TYPE DROPDOWN (attempting to open...)")
        print("=" * 80)
        # Try multiple ways to open chart type menu
        for sel in ['[data-name="chart-style-button"]', '#header-toolbar-chart-styles',
                    'button[aria-label*="Chart type"]', 'button[aria-label*="Chart Type"]',
                    'button[aria-label*="chart style"]']:
            try:
                btn = page.locator(sel).first
                if await btn.count() > 0 and await btn.is_visible():
                    await btn.click()
                    print(f"  Opened via: {sel}")
                    break
            except Exception:
                continue
        else:
            print("  Could not auto-open chart type dropdown")
            gui_prompt("Please click the chart type button (candle icon in toolbar).\n\nClick OK when dropdown is open.")

        await asyncio.sleep(1)

        # Dump dropdown/menu items
        menu_items = await page.evaluate("""() => {
            // Look for any visible menu/dropdown/popup
            const containers = document.querySelectorAll(
                '[class*="menu"], [class*="popup"], [class*="dropdown"], [role="menu"], [role="listbox"]'
            );
            const items = [];
            containers.forEach(c => {
                if (c.offsetParent === null) return;  // skip hidden
                c.querySelectorAll('[role="option"], [role="menuitem"], [class*="item"], div, span').forEach(el => {
                    const text = el.textContent?.trim();
                    if (text && text.length > 1 && text.length < 40 && el.children.length < 3) {
                        items.push({
                            text: text,
                            tag: el.tagName,
                            role: el.getAttribute('role'),
                            dataName: el.getAttribute('data-name'),
                            className: el.className?.substring(0, 60),
                        });
                    }
                });
            });
            return items;
        }""")
        seen = set()
        for item in menu_items:
            key = item['text']
            if key in seen:
                continue
            seen.add(key)
            parts = [f"text='{key}'"]
            if item.get("role"): parts.append(f"role={item['role']}")
            if item.get("dataName"): parts.append(f"data-name={item['dataName']}")
            print(f"  {' | '.join(parts)}")

        # Close dropdown
        await page.keyboard.press("Escape")
        await asyncio.sleep(0.5)

        # ── Try to find the hamburger/main menu ──
        print("\n" + "=" * 80)
        print("MAIN MENU (attempting to open...)")
        print("=" * 80)
        for sel in ['button[aria-label="Open menu"]', 'button[aria-label="Main menu"]',
                    '[data-name="save-load-menu-button"]', '[data-name="main-menu-button"]',
                    'button[data-name="burger-menu"]', '#header-toolbar-save-load']:
            try:
                btn = page.locator(sel).first
                if await btn.count() > 0 and await btn.is_visible():
                    await btn.click()
                    print(f"  Opened via: {sel}")
                    break
            except Exception:
                continue
        else:
            print("  Could not auto-open main menu")
            gui_prompt("Please click the hamburger menu (top-left).\n\nClick OK when menu is open.")

        await asyncio.sleep(1)

        # Dump menu items
        menu_items2 = await page.evaluate("""() => {
            const containers = document.querySelectorAll(
                '[class*="menu"], [class*="popup"], [class*="dropdown"], [role="menu"]'
            );
            const items = [];
            containers.forEach(c => {
                if (c.offsetParent === null) return;
                c.querySelectorAll('[role="menuitem"], [class*="item"], div, span, a').forEach(el => {
                    const text = el.textContent?.trim();
                    if (text && text.length > 2 && text.length < 60 && el.children.length < 3) {
                        items.push({
                            text: text,
                            tag: el.tagName,
                            role: el.getAttribute('role'),
                            dataName: el.getAttribute('data-name'),
                        });
                    }
                });
            });
            return items;
        }""")
        seen2 = set()
        for item in menu_items2:
            key = item['text']
            if key in seen2:
                continue
            seen2.add(key)
            parts = [f"text='{key}'"]
            if item.get("role"): parts.append(f"role={item['role']}")
            if item.get("dataName"): parts.append(f"data-name={item['dataName']}")
            print(f"  {' | '.join(parts)}")

        await page.keyboard.press("Escape")

        print("\n" + "=" * 80)
        print("DISCOVERY COMPLETE — copy the output above for selector debugging")
        print("=" * 80)

        gui_prompt("Discovery complete!\n\nCheck the terminal output for selector info.\nClick OK to close browser.")
        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
