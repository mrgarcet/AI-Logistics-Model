#----------------------------------------------------
#  Path: src/app/main.py
#----------------------------------------------------
'''
Unified CLI: type natural language or `scan` to start barcode loop.
Run: python -m app.main
'''

from .chatbot import handle_user_query
from src.cv.barcode import webcam_mode          # reuse the scanner

def main():
    print("Inventory AI – type a request or `scan` (ESC to quit scanner)")
    while True:
        try:
            user = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user: continue
        if user.lower() == "scan":
            webcam_mode()                       # updates inventory.csv
            continue
        print(handle_user_query(user))

if __name__ == "__main__":
    main()

## Additional Information ##
## from the project root
# .\.venv\Scripts\Activate
# to run main.py use:
''' python -m src.app.main '''