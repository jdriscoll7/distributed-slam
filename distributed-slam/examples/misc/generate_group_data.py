from utility.data_generation import create_random_grouped_dataset


if __name__ == "__main__":
    graph = create_random_grouped_dataset(0.1, 0.1, [5, 10, 15], "grouped_set")
    print("")
