import os
import sys


def get_data_path(app_name: str) -> str:
    if sys.platform == "win32":
        result_path = os.path.join(os.environ["AppData"], app_name)
    elif sys.platform == "darwin":
        result_path = os.path.join(os.path.expanduser("~"), "Library", "Application Support", app_name)
    else:
        result_path = os.path.join(os.path.expanduser("~"), ".config", app_name)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    return result_path
