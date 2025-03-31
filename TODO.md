Experimental Setup:

1. Train all three models on 10% of the data (ca. 5000)
- fine tune every model as far as possible, max. 5 attempts.
2. Select the two better performing ones
- select the two best vectorization types
- select the 4 best tokenization methods
3. Train the two models on 30% of the data (ca. 15000)
- fine tune every model as far as possible, max. 5 attempts.
4. Select the best performing model
- fine tune the best performing model
- select the best vectorization method
- select the 2 best tokenization methods
5. Select the best method at all.


Use for all runs different random states