# libraries
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

#data
housing_file_path = '/Housing.csv'
housing_data = pd.read_csv(housing_file_path)

# preparing model values
y = housing_data.price
features = ['area', 'bedrooms','bathrooms','stories','parking']
x = housing_data[features]
train_x, val_x, train_y, val_y = train_test_split(x, y, random_state = 0)

# getting the margin of error
def get_mae(max_leaf_nodes, train_x, val_x, train_y, val_y):
    mae_test_model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    mae_test_model.fit(train_x, train_y)
    mae_test_model_predictions = mae_test_model.predict(val_x)
    mae = mean_absolute_error(val_y, mae_test_model_predictions)
    return(mae)


# finding most accurate leaf node by finding the smallest margin of error when making a model with different numbers of leaf nodes
r1 = 5
r2 = 500
max_leaf_nodes_options = list(range(r1,r2+1))
margin_of_errors = [get_mae(i, train_x, val_x, train_y, val_y) for i in max_leaf_nodes_options]

best_tree_size = max_leaf_nodes_options[margin_of_errors.index(min(margin_of_errors))]

# creating final model
housing_machine_model = DecisionTreeRegressor(max_leaf_nodes=best_tree_size, random_state=0)
housing_machine_model.fit(train_x, train_y)
housing_machine_model_predictions = housing_machine_model.predict(val_x.head())

# printing data
print(f"Housing data: \n{housing_data.head()}")
print("\n\n")
print(f" The predicted prices of the houses in the data are:\n")
z=0
for i in housing_machine_model_predictions:
    print(f"{z} {i}\n") 
    z=z+1

print(f"\nMargin of Error: {min(margin_of_errors)}")