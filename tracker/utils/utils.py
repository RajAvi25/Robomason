import os

class Utils:
    @staticmethod
    def get_next_folder(base_path):
        existing_folders = [
            d for d in os.listdir(base_path)
            if os.path.isdir(os.path.join(base_path, d)) and d.startswith("test")
        ]
        folder_numbers = [
            int(folder.split(" ")[1])
            for folder in existing_folders if folder.split(" ")[1].isdigit()
        ]
        next_number = max(folder_numbers, default=0) + 1
        return os.path.join(base_path, f"test {next_number}")
