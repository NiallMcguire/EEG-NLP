
import os




if __name__ == "__main__":
    config_save_path = "/users/gxb18167/configs/"


    # print files in the directory
    print("Files in the directory:")
    for file in os.listdir(config_save_path):
        print(file)
