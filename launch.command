#!/usr/bin/env bash
# macOS double-click launcher for Project Third-Eye
cd "$(dirname "$0")" || exit 1

if [ ! -d ".venv" ]; then
    echo "Virtual environment not found. Please create .venv first."
    read -n 1 -s -r -p "Press any key to exit..."
    exit 1
fi

echo "Activating Python virtual environment (.venv)..."
source .venv/bin/activate

echo "Launching Project Third-Eye Streamlit Interface..."
exec streamlit run app/main.py
