from collections import OrderedDict


def broadcast_server_to_client_initialization(server_state_dict, client_mask, client_state_dict):
    """
    mask=0: shared parameter; overwrite with the server value
    mask=1: personalized parameter; retain the client value
    """
    client_new = OrderedDict()
    for key in client_state_dict.keys():
        if key not in client_mask:
            client_new[key] = client_state_dict[key].clone()
            continue

        assert client_mask[key].shape == client_state_dict[key].shape, f"Mask shape mismatch for {key}"
        assert server_state_dict[key].shape == client_state_dict[key].shape, f"Server/client shape mismatch for {key}"

        # Get the target device.
        device = client_state_dict[key].device

        client_new[key] = client_state_dict[key].clone()

        # Ensure the mask is on the correct device.
        mask_tensor = client_mask[key]
        if mask_tensor.device != device:
            mask_tensor = mask_tensor.to(device)
        global_idx = (mask_tensor == 0)

        # Ensure server_state_dict[key] is on the correct device.
        server_tensor = server_state_dict[key]
        if server_tensor.device != device:
            server_tensor = server_tensor.to(device)

        client_new[key][global_idx] = server_tensor[global_idx]
    return client_new