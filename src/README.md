## Python - Using Keras
Raghav

### Setup and Installation (Requires Conda and pip)
1. Setup a conda environment to manage all the python dependencies needed to run this project
    - For example, `conda create -n league-ai python=3.6.5` and then `conda activate league-ai`
2. `cd` to this repo, into the src directory: `cd ..league-ai/src`
3. Run `pip install -r requirements.txt` to install all the Python dependencies
4. Run the app:
    - `python app.py` - will assume the fixed run mode, where the training, test, and devlopment data have already been partitioned. Preprocess those.
    <!-- - `python app.py` - will ranfomly use 80% of the 2017, 2018, and 2019 data to train the models, and the other 20% to test the models. At the end you will see the results for the model that performed the best. -->

### App Workflow

Steps:
1. Fetch datasets (CSVS)
2. Inspect and Clean dataset
3. Build models
4. Train model on training data
5. Test model on development set
6. Predict on the test set


#### Data
- We are using 2017, 2018, and 2019 LCS Data (Spring, Summer, and Worlds). Each dataset
is stored in its own CSV file.
- data.py, regardless of method, returns a tuple of training data, training labels, test data, test labels (each as a numpy array)
    - Method 1 - Random - (This is turned off for now)
        <!-- - reads all the CSVs and merges them together
        - filter data:
            - only keeps rows pertaining to teams (not individual players)
            - only keeps the columns "side", "goldat15", "totalgold", "gdat10", "xpdat10", "gdat15", "fb", "firsttothreetowers", "result"
            - 
            - replaced Map Side "Red" and "Blue" as numbers 2 and 1
            - ignores rows with null values for certain stats        
        - splits the data randomly into two sets:
            - 80% data for training (this is additionally split into data and labels)
            - 20% data for testing  (this is additionally split into data and labels) -->
    - Method 2 - Fixed
        <!-- - reads all 2017 and 2018 CSVs and merges them together into one dataframe (training dataframe)
        - reads 2019 data into another dataframe (test dataframe) -->
        - 2017, 2018, and 2019 data has been split into training, development, and test data sets.
        - filter each data frame like so:
            - only keeps rows pertaining to teams (not individual players)
            - only keeps the columns "side", "gdat10", "xpdat10", "gdat15", "fb", "firsttothreetowers", "result"
            - replaced Map Side "Red" and "Blue" as numbers 2 and 1
            - ignores rows with null values for certain stats        
        - splits each data frame into data alone and corresponding labels
    
 #### ML Models
 - Build and compile each model according to some configuration
    - The configuration will specify things like whether to build a neural network, how many layers, how many nodes per layer, what 
    kind of activation function, what optimizer to use.
    - model.py will return all the compiled models in a list that also contains other metadata used to construct the models.
    
 - Train the model (give it training data and labels)
 - Evaluate the model (give it test data and test labels) to test is accuracy
 
 - Use the model now to predict.
