from streamlit.testing.v1 import AppTest

def test_app_start():
    AppTest.from_file("app.py").run(timeout=30) 
    