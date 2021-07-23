## What are in each dataset folder 

Each dataset folder contains four files:

```
/
├── feature_dict.pkl       --> (A dictionary that contains all attribute name with corresponding feature index)
├── implicit_ratings.csv   --> (A csv file containing ratings of each data sample)
├── item_dict.pkl          --> (A dictionary containing the attributes of each item)
├── user_dict.pkl          --> (A dictionary containing the attributes of each user)
```

## To run our code on your own datasets, generate files using the following format:

```
feature_dict.pkl:
Format: {attributeName: index}
```

```
implicit_ratings.csv
Format: user_id, item_id, rating
```

```
item_dict.pkl
Format: item_id: {'title': item index, 'attribute': a list of attributes}  
```

```
user_dict.pkl
Format: item_id: {'name': user index, 'attribute': a list of attributes}  
```
