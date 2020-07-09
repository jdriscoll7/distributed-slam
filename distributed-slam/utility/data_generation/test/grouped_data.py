from utility.data_generation import create_random_grouped_dataset
from utility.parsing import parse_g2o

if __name__ == "__main__":

    create_random_grouped_dataset(2, 3, [5, 10, 15], 'grouped_data.g2o')
    a = parse_g2o('grouped_data.g2o', groups=True)
    print('')