import numpy as np

from edop import phenotype


TEST_POPULATIONS = (
    # (basal, alpha, hill), expected result
    ((0.6, 1, 2), 1.1834),

)


if __name__ == '__main__':
    for (basal, alpha, hill), expected_result in TEST_POPULATIONS:
        population = np.empty(1, dtype=[
            ('basal', '<f8'), ('alpha', '<f8'), ('hill', '<f8')
        ])
        population['basal'] = basal
        population['alpha'] = alpha
        population['hill'] = hill

        test_result = phenotype(population)
        np.testing.assert_allclose(test_result, expected_result, atol=1e-4)
