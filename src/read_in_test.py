import data



if __name__ == "__main__":
    task_name = "task1, task2, taskNRv2"

    data = data.Data()
    data.create_custom_dataset(task_name)

    print("Data created successfully!")