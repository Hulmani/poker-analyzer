import pyautogui
import time

print("Screen size:", pyautogui.size())
print("Move mouse slowly around scrcpy window...")

try:
    while True:
        x, y = pyautogui.position()
        # Only print if coordinates are reasonable
        if 0 <= x <= 1920 and 0 <= y <= 1080:
            print(f"Mouse: {x}, {y}        ", end='\r')
        time.sleep(0.1)
except KeyboardInterrupt:
    print(f"\nFinal position: {x}, {y}")