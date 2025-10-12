from ofdm_based_systems.simulation.models import Simulation


def main():
    simulation = Simulation(
        num_bits=None,
        num_symbols=10_000_000,
        num_subcarriers=64,
    )

    simulation.run()


if __name__ == "__main__":
    main()
