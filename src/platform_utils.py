import os
import sys


def is_package_available(package_name: str, min_version: str = None) -> bool:
    import importlib

    try:
        module = importlib.import_module(package_name)
        if min_version is None:
            return True

        MAX_COMPARISON_LENGTH = 3
        versions = str(module.__version__).split(".")[:MAX_COMPARISON_LENGTH]
        min_versions = min_version.split(".")[:MAX_COMPARISON_LENGTH]

        for i in range(MAX_COMPARISON_LENGTH):
            current = int(versions[i]) if i < len(versions) else 0
            check = int(min_versions[i]) if i < len(min_versions) else 0
            if current > check:
                return True
            elif current < check:
                return False
        return True
    except (ImportError, AttributeError, ModuleNotFoundError, ValueError):
        pass
    return False


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
