from synthetic_AB_tests.synthetic import Synthetic_AA_test


def main():
    my_test = Synthetic_AA_test(
        n_actions=10000,
        gauss_noise=0.2
    )

    test, control = my_test.generate_test_control_groups(
        test_size=1000,
        control_size=1000,
        metric_distribution="normal"
    )

    my_test.visualize_groups()


if __name__ == "__main__":
    main()