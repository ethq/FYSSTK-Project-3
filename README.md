# FYSSTK-Project-3

To create a dataset, use create_xy_data.py. Saves to Data/

To analyze the dataset, use the (monstrous) state machine Analyze_XY.py
Upon initialization with a dataset, it will 
    1) Create/load a design matrix
    2) Do a polynomial fit to the averaged energy(temperature) curve
    3) Calculate the heat capacity and the critical temperature from it

Plots will be shown if the corresponding flag is set, but they are always saved.

Training is done through .train_nn(). For a list of possible models, see get_model.py. Models are saved on validation accuracy checkpoints to Models/ for re-use. Saves plots of training history and the confusion matrix.

To find the critical temperature predicted by a model, use .tkt_from_pred(). If a pre-trained model exists, it will be loaded. Saves plots of prediction probabilities for states at any given temperature.


The (16, 5000, 1) and (32, 1000, 1) datasets have not been uploaded, as they are about 250 and 500 MB in size respectively.
