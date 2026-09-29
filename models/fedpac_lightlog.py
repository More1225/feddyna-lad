"""
FedPAC-specific LightLog model

This model extends the base LightLog model with features required by FedPAC:
1. Supports feature extraction
2. Explicitly identifies classifier-head parameters
3. Remains compatible with the existing interface

This model is used only for FedPAC experiments; other methods are unaffected.
"""

import torch
import torch.nn as nn

from models.lightlog import LightLogModel as BaseLightLogModel
from models.lightlog import TCNBackbone as BaseTCNBackbone


class FedPACTask():
    """FedPAC task metadata container"""
    def __init__(self, num_classes=2):
        self.num_classes = num_classes


class FedPACTCNBackbone(BaseTCNBackbone):
    """
    FedPAC-specific TCN backbone

    Override forward to optionally return both features and logits.
    """

    def forward(self, x, return_features=False):
        """
        Run the forward pass.

        Args:
            x: Input tensor with shape (B, seq_len, input_dim).
            return_features: Whether to return features.

        Returns:
            When return_features=True, return (features, logits).
            Otherwise, return logits.
        """
        # x: (B, seq_len, input_dim)
        x = self.compress(x)         # (B, seq_len, compress_dim) or unchanged
        x = x.permute(0, 2, 1)       # (B, C, seq_len)

        for block in self.blocks:
            x = block(x)

        features = self.pool(x).squeeze(-1)  # (B, feature_dim)

        if return_features:
            # Return features and logits.
            features_drop = self.head_dropout(features)
            logits = self.fc(features_drop)
            return features, logits
        else:
            # Return logits only (preserves the original behavior).
            x = self.head_dropout(features)
            x = self.fc(x)
            return x

    def extract_features(self, x):
        """Extract features without computing logits."""
        # x: (B, seq_len, input_dim)
        x = self.compress(x)
        x = x.permute(0, 2, 1)

        for block in self.blocks:
            x = block(x)

        features = self.pool(x).squeeze(-1)
        return features


class FedPACLightLogModel(BaseLightLogModel):
    """
    FedPAC-specific LightLog model

    Extend the base model with FedPAC-specific functionality.
    """

    def __init__(
        self,
        num_classes=2,
        filters_list=None,
        input_dim=300,
        dropout=0.3,
        compress_dim=96
    ):
        # Use the FedPAC-specific TCNBackbone.
        tcn_backbone = FedPACTCNBackbone(
            input_dim=input_dim,
            num_classes=num_classes,
            filters_list=filters_list,
            dropout=dropout,
            compress_dim=compress_dim
        )

        # Construct the model manually to avoid calling the parent __init__.
        super(BaseLightLogModel, self).__init__()  # Call nn.Module.__init__.
        self.tcn = tcn_backbone

        # FedPAC-specific attributes
        self.num_classes = num_classes
        self.filters_list = filters_list if filters_list is not None else [32, 16, 8]
        self.input_dim = input_dim
        self.dropout = dropout
        self.compress_dim = compress_dim

        # Explicitly identify classifier-head parameter names for FedPAC.
        self.classifier_weight_keys = ['tcn.fc.weight', 'tcn.fc.bias']
        self.base_weight_keys = []

        # Collect all parameter names.
        for name, _ in self.named_parameters():
            if name not in self.classifier_weight_keys:
                self.base_weight_keys.append(name)

    def forward(self, x, return_features=False):
        """
        Run the forward pass.

        Args:
            x: Input tensor.
            return_features: Whether to return features.

        Returns:
            When return_features=True, return (features, logits).
            Otherwise, return logits.
        """
        return self.tcn(x, return_features)

    def extract_features(self, x):
        """Extract features."""
        return self.tcn.extract_features(x)

    def feature2logit(self, x):
        """Convert features to logits."""
        features_drop = self.tcn.head_dropout(x)
        logits = self.tcn.fc(features_drop)
        return logits

    def get_classifier_keys(self):
        """Get classifier-head parameter names for FedPAC."""
        return self.classifier_weight_keys

    def get_feature_dim(self):
        """Get the feature dimension."""
        if self.tcn.filters_list:
            return self.tcn.filters_list[-1]  # Number of channels in the final convolution layer.
        elif self.tcn.compress_dim:
            return self.tcn.compress_dim
        else:
            return self.tcn.input_dim


def create_fedpac_model(args):
    """
    Create a FedPAC-specific model.

    Args:
        args: configuration arguments

    Returns:
        FedPAC-specific model
    """
    return FedPACLightLogModel(
        num_classes=args.num_classes,
        filters_list=args.filters_list,
        input_dim=args.input_dim,
        dropout=args.dropout,
        compress_dim=args.compress_dim
    )


# Manual smoke test
def test_fedpac_model():
    """Test the FedPAC-specific model"""
    import sys

    class MockArgs:
        def __init__(self):
            self.num_classes = 2
            self.filters_list = [32, 16, 8]
            self.input_dim = 300
            self.dropout = 0.3
            self.compress_dim = 96

    args = MockArgs()

    # Create the model.
    model = create_fedpac_model(args)

    # Create test input.
    x = torch.randn(4, 75, args.input_dim)

    # Check the default call (expected to return logits).
    logits = model(x)
    print(f"Default-call output shape: {logits.shape}")  # Expected: (4, 2)
    assert logits.shape == (4, 2), f"Unexpected default-call output shape: {logits.shape}"

    # Check feature extraction.
    features, logits2 = model(x, return_features=True)
    print(f"Feature-extraction output shapes: features= {features.shape},  logits= {logits2.shape}")
    assert features.shape == (4, model.get_feature_dim()), f"Unexpected feature shape: {features.shape}"
    assert logits2.shape == (4, 2), f"Unexpected logits shape: {logits2.shape}"

    # Check standalone feature extraction.
    features2 = model.extract_features(x)
    print(f"Standalone feature shape: {features2.shape}")
    assert torch.allclose(features, features2), "Feature extraction results do not match."

    # Check feature-to-logit conversion.
    logits3 = model.feature2logit(features)
    print(f"Feature-to-logit output shape: {logits3.shape}")
    assert logits3.shape == (4, 2), f"Unexpected converted-feature shape: {logits3.shape}"

    # Check classifier-head parameter identification.
    classifier_keys = model.get_classifier_keys()
    print(f"Classifier-head parameters: {classifier_keys}")
    assert 'tcn.fc.weight' in classifier_keys, "Classifier weight is missing."
    assert 'tcn.fc.bias' in classifier_keys, "Classifier bias is missing."

    print("✓ FedPAC-specific model test passed")
    return True


if __name__ == "__main__":
    test_fedpac_model()