# kaggle-whats-cooking
Solution for [Kaggle whats-cooking problem](https://www.kaggle.com/c/whats-cooking) based on random forest classifier algorithm

## Usage
### Requirements installation
```Bash
pip install -r requirements.txt
```

### Running script
You can get usage help using -h key
```Bash
python main.py -h
```

### Verify model
To test model locally you can run script in verification mode providing two data files in JSON format: training and verification
```Bash
python main.py verify -t <train data JSON file> -v <verification data JSON file>
```

### Evaluate model
To run model on unmarked dataset you can run script in evaluation mode providing three arguments: path to trainig data file, path to unmarked data file and output file path
```Bash
python main.py evaluate -t <train data file> -e <data to run model on> -o <output csv file>
```

## Results
This solution was tested on datasets provided by Kaggle, so they can be used for testing this algorithm. Kaggle scored this solution with 0.72938 points.
