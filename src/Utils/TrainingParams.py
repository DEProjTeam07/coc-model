def validate_params(epochs, learning_rate):
    if not (1 <= epochs <= 100):
        raise ValueError("Epochs must be between 1 and 100.")
    if not (1e-5 <= learning_rate <= 1):
        raise ValueError("Learning rate must be between 0.00001 and 1.")
