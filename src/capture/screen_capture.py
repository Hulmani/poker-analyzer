import pyautogui
try:
    screenshot = pyautogui.screenshot(region=(825, 77, 418, 918))
    screenshot.save("test_capture.png")
    print("Test capture successful - check test_capture.png")
except Exception as e:
    print(f"Test capture failed: {e}")