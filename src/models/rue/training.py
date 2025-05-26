import time
from src.models.misc import set_seed
from src.models.rue.model import AE_Regressor


def model_training(
    hp_dict, predictors, pred_cols, train_df, valid_df, seed,
    batch_size=128, max_epochs=5000, verbose=1, patience=10):
    set_seed(seed)

    # Get data
    train_X, train_y = (
        train_df[predictors].values.astype('float32'), train_df[pred_cols].values.astype('float32'))
    valid_X, valid_y = (
        valid_df[predictors].values.astype('float32'), valid_df[pred_cols].values.astype('float32'))

    # Train Regressor
    ae_regressor = AE_Regressor(**hp_dict, predictors=predictors, output_features=pred_cols)
    valid_loss_regressor, best_epoch = ae_regressor.train_regressor(
        train_X, train_y, valid_X, valid_y,
        batch_size, max_epochs, verbose, patience)

    # Train decoder
    valid_loss_ae = ae_regressor.train_decoder(
        train_X, valid_X, batch_size, max_epochs, verbose, patience)
    return ae_regressor


def model_training_predictor(
    hp_dict, predictors, pred_cols, train_df, valid_df, seed,
    batch_size=128, max_epochs=5000, verbose=1, patience=10):

    import time
    set_seed(seed)

    # Get data
    train_X, train_y = (
        train_df[predictors].values.astype('float32'), train_df[pred_cols].values.astype('float32'))
    valid_X, valid_y = (
        valid_df[predictors].values.astype('float32'), valid_df[pred_cols].values.astype('float32'))

    # Train Regressor
    ae_regressor = AE_Regressor(**hp_dict, predictors=predictors, output_features=pred_cols)

    start = time.time()
    # Train classifier
    valid_loss_regressor, best_epoch = ae_regressor.train_regressor(
        train_X, train_y, valid_X, valid_y,
        batch_size, max_epochs, verbose, patience)

    # Train decoder
    # valid_loss_ae = ae_regressor.train_decoder(
    #     train_X, valid_X, batch_size, max_epochs, verbose, patience)

    print(f"Training took {time.time()-start}s.")
    return ae_regressor


def model_training_decoder(
    hp_dict, predictors, pred_cols, train_df, valid_df, seed, prev_model,
    batch_size=128, max_epochs=5000, verbose=1, patience=10):
    set_seed(seed)

    # Get data
    train_X, train_y = (
        train_df[predictors].values.astype('float32'), train_df[pred_cols].values.astype('float32'))
    valid_X, valid_y = (
        valid_df[predictors].values.astype('float32'), valid_df[pred_cols].values.astype('float32'))

    # Train Regressor
    ae_regressor = AE_Regressor(**hp_dict, predictors=predictors, output_features=pred_cols)
    ae_regressor.replace_encoder_predictor(prev_model)

    start = time.time()
    # Train classifier
    # valid_loss_regressor, best_epoch = ae_regressor.train_regressor(
    #     train_X, train_y, valid_X, valid_y, 
    #     batch_size, max_epochs, verbose, patience)

    # Train decoder
    valid_loss_ae = ae_regressor.train_decoder(
        train_X, valid_X, batch_size, max_epochs, verbose, patience)

    print(f"Training took {time.time()-start}s.")
    return ae_regressor