import os

def check_model_directory():
    model_dir = "models/"
    if not os.path.exists(model_dir):
        print("Models directory does not exist!")
        return
    
    files = os.listdir(model_dir)
    if len(files) == 0:
        print("No model weights found in the models/ directory.")
    else:
        print("Found the following model files:")
        for file in files:
            print(file)

if __name__ == "__main__":
    check_model_directory()
