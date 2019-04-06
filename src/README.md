### Python - Using Keras
Raghav

Steps:
1. Fetch datasets (CSVS)
2. Inspect and Clean dataset
3. Build Keras model 
4. Train model
5. Test model


#### Data
- We are using 2018 LCS Data (Spring, Summer, and Worlds). Each dataset
is stored in its own CSV file.
- data.py programmatically 
    - reads all the CSVs and merges them together
    - filter data:
        - only keeps rows pertaining to teams (not individual players)
        - only keeps the columns "side", "gdat10", "xpdat10", "gdat15", "fb", "firsttothreetowers", "result"
        - replaced Map Side "Red" and "Blue" as numbers 2 and 1
        - ignores rows with null values for certain stats        
    - splits the data randomly into two sets:
        - 80% data for training (this is additionally split into data and labels)
        - 20% data for testing  (this is additionally split into data and labels)
    - returns a tuple of training data, training labels, test data, test labels (each as a numpy array)
    
 #### Keras
 - Create a Keras Sequential model - neural network:
    - 6 features
    - 2 Hidden layers, one with 128 nodes, the other with 10
    
 - Train the model (give it training data and labels)
 - Evaluate the model (give it test data and test labels) to test is accuracy
 
 - Use the model now to predict.
 
 #### Observatiions
 - Since I split the data randomly into test data and training data
 sometimes we have 75% accuracy and sometimes we have 30%.
