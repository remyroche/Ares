from keras import backend as K


def create_pnl_aware_loss(
    pnl_multiplier=0.1, liquidation_penalty=2.0, reward_boost=1.5
):
    """
    This is a factory function that creates a custom Keras loss function.
    It combines standard classification loss (cross-entropy) with a financial
    component that heavily penalizes high-risk errors and rewards high-profit
    correct predictions, teaching the model to prioritize capital preservation.
    """

    def pnl_aware_loss(y_true, y_pred):
        """
        Calculates the combined loss.

        Args:
            y_true: Ground truth tensor with shape (batch_size, num_classes + 2).
                    It contains [one_hot_label, reward_potential, risk_potential].
            y_pred: Predicted probabilities with shape (batch_size, num_classes).
        """
        # --- Unpack the ground truth tensor ---
        y_true_labels = y_true[:, :-2]
        # num_classes = tf.shape(y_true_labels)[1] # Removed: F841 - local variable assigned but never used

        # The last two elements are the financial outcomes
        reward_potential = y_true[:, -2]
        risk_potential = y_true[:, -1]  # This is the distance to liquidation

        # --- 1. Standard Classification Loss ---
        ce_loss = K.categorical_crossentropy(y_true_labels, y_pred)

        # --- 2. Financial (PnL) Loss Component ---
        # Get the model's confidence in the correct prediction
        true_class_probs = K.sum(y_true_labels * y_pred, axis=-1)

        # Get the model's confidence in its highest-probability (potentially wrong) prediction
        # predicted_class_probs = K.max(y_pred, axis=-1) # Removed: F841 - local variable assigned but never used

        # Identify when the model's prediction is wrong
        is_wrong = 1.0 - K.cast(
            K.equal(K.argmax(y_true_labels), K.argmax(y_pred)), dtype="float32"
        )

        # Calculate the financial loss:
        # - If correct, we get a "negative loss" (a reward) proportional to the profit potential.
        # - If wrong, we get a large penalty proportional to the liquidation risk.
        financial_loss = (1.0 - true_class_probs) * (
            risk_potential * liquidation_penalty
        ) * is_wrong - (true_class_probs * reward_potential * reward_boost) * (
            1.0 - is_wrong
        )

        # --- 3. Combine the Losses ---
        combined_loss = ce_loss + (financial_loss * pnl_multiplier)

        return combined_loss

    return pnl_aware_loss
