In order to evaluate the outputs of your model, make sure your entries are saved in a ``.json`` that follows the format below:

```
{
    "english": natural language specification of the task.make sure this follows exactly the same template as the synthesized data. This should be the same as the 'nl_description' of the corresponding entry in your test set.
    "ground_truth": the ground truth plan generated during the data synthesis process.
    "generated": The plan produced by your model. 
} 
```

In order to get metrics for the model's outputs on the dataset run the following:

**Single Goal**:

``python executor-point-sg.py $path_to_test_data $path_to_model_outputs``

**Multi-Goal**:

``python executor-mg.py $path_to_test_data $path_to_model_outputs``