#once the data has been created, use the code below to load it and split it into training/testing/validation sets:

#load
import pandas as pd

total_input_load = pd.read_csv("total_input.csv")
total_input_load = total_input_load.as_matrix()
total_input_load = np.delete(total_input_load, 0, 1)
print(total_input_load.shape)

total_output_load = pd.read_csv("total_output.csv")
total_output_load = total_output_load.as_matrix()
total_output_load = np.delete(total_output_load, 0, 1)
print(total_output_load.shape)

condensed_output_load = pd.read_csv("condensed_output.csv")
condensed_output_load = condensed_output_load.as_matrix()
condensed_output_load = np.delete(condensed_output_load, 0, 1)
print(condensed_output_load.shape)

total_digit_count_load = pd.read_csv("total_digit_count.csv")
total_digit_count_load = total_digit_count_load.as_matrix()
total_digit_count_load = np.delete(total_digit_count_load, 0, 1)
print(total_digit_count_load.shape)

total_location_load = pd.read_csv("total_location.csv")
total_location_load = total_location_load.as_matrix()
total_location_load = np.delete(total_location_load, 0, 1)
print(total_location_load.shape)

is_it_a_digit_load = pd.read_csv("is_it_a_digit.csv")
is_it_a_digit_load = is_it_a_digit_load.as_matrix()
is_it_a_digit_load = np.delete(is_it_a_digit_load, 0, 1)
print(is_it_a_digit_load.shape)

is_it_a_digit_label_load = pd.read_csv("is_it_a_digit_label.csv")
is_it_a_digit_label_load = is_it_a_digit_label_load.as_matrix()
is_it_a_digit_label_load = np.delete(is_it_a_digit_label_load, 0, 1)
print(is_it_a_digit_label_load.shape)

#in total_location_load, remove unnecessary data.
amended_total_location_load = removeExtraLocation(total_location_load, total_digit_count_load)

#split the data into training and testing sets
condensed_X_train, condensed_X_test, condensed_y_train, condensed_y_test = train_test_split(total_input_load, condensed_output_load, test_size = 0.25, random_state = 7)

full_X_train, full_X_test, full_y_train, full_y_test = train_test_split(total_input_load, total_output_load, test_size = 0.25, random_state = 7)
count_X_train, count_X_test, count_y_train, count_y_test = train_test_split(total_input_load, total_digit_count_load, test_size = 0.25, random_state = 7)
loc_X_train, loc_X_test, loc_y_train, loc_y_test = train_test_split(total_input_load, amended_total_location_load, test_size = 0.25, random_state = 7)
