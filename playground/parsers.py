from autoencoder import MLPEncoder, MLPDecoder
from fvt_encoder import FvTEncoder
from black_box_network import BlackBoxNetwork


def parse_encoder(config: dict) -> BlackBoxNetwork:
    assert "type" in config
    if config["type"] == "FvTEncoder":
        assert "dim_input_jet_features" in config
        assert "dim_intermed_dijet_features" in config
        assert "dim_intermed_quadjet_features" in config
        assert "dim_output" in config

        dim_input_jet_features = config["dim_input_jet_features"]
        dim_intermed_dijet_features = config["dim_intermed_dijet_features"]
        dim_intermed_quadjet_features = config["dim_intermed_quadjet_features"]
        dim_output = config["dim_output"]

        return FvTEncoder(
            dim_input_jet_features,
            dim_intermed_dijet_features,
            dim_intermed_quadjet_features,
            dim_output,
        )

    elif config["type"] == "MLPEncoder":
        assert "input_dim" in config
        assert "latent_dim" in config
        assert "hidden_dims" in config

        input_dim = config["input_dim"]
        latent_dim = config["latent_dim"]
        hidden_dims = config["hidden_dims"]
        activation = config.get("activation", "SiLU")
        last_bias = config.get("last_bias", False)

        return MLPEncoder(input_dim, latent_dim, hidden_dims, activation, last_bias)
    else:
        raise NotImplementedError(f"Encoder type {config['type']} not implemented")


def parse_decoder(config: dict) -> BlackBoxNetwork:
    assert "type" in config
    if config["type"] == "MLPDecoder":
        assert "latent_dim" in config
        assert "output_dim" in config
        assert "hidden_dims" in config

        latent_dim = config["latent_dim"]
        output_dim = config["output_dim"]
        hidden_dims = config["hidden_dims"]
        activation = config.get("activation", "SiLU")
        last_bias = config.get("last_bias", False)

        return MLPDecoder(latent_dim, output_dim, hidden_dims, activation, last_bias)
    else:
        raise NotImplementedError(f"Decoder type {config['type']} not implemented")
