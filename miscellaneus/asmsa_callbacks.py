import tensorflow as tf
import asmsa.visualizer as visualizer

def callbacks(log_dir, model, inputs, freq=10, monitor="val_loss"):
    cb = [
        tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch',
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=15,
            min_delta=1e-6,
            restore_best_weights=True,
            verbose=1,
            mode="min"
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=0.5,
            patience=7,
            min_lr=1e-7,
            verbose=1
        ),
        visualizer.VisualizeCallback(model,freq=freq,inputs=inputs,figsize=(12,3))]
    return cb
    