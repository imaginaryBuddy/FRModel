import unittest

import pandas as pd
from tqdm import tqdm


from frmodel.express.classification.classification import Classification
import numpy as np

class TestClassification(unittest.TestCase):

    def test_classify(self):

        f = Classification()

        TRIALS = 15
        EXPERIMENTS = 8
        ESTIMATORS = [5, 10, 15]
        DEPTHS = [1, 2, 3, 4, 5, None]

        results = np.zeros([EXPERIMENTS, len(ESTIMATORS), len(DEPTHS), TRIALS])

        for trial in tqdm(range(TRIALS)):
            for est_i, est in enumerate(ESTIMATORS):
                for dep_i, dep in enumerate(DEPTHS):
                    f.repartition()
                    f.repartition_fake()
                    results[0, est_i, dep_i, trial] = f.m_m(est, dep)[0]
                    results[1, est_i, dep_i, trial] = f.d_d(est, dep)[0]
                    results[2, est_i, dep_i, trial] = f.md_md(est, dep)[0]
                    results[3, est_i, dep_i, trial] = f.m_d(est, dep)[0]
                    results[4, est_i, dep_i, trial] = f.d_m(est, dep)[0]
                    results[5, est_i, dep_i, trial] = f.mc_mc(est, dep)[0]
                    results[6, est_i, dep_i, trial] = f.dc_dc(est, dep)[0]
                    results[7, est_i, dep_i, trial] = f.mdc_mdc(est, dep)[0]

        result = np.mean(results, axis=-1).reshape([8, -1])
        pd.DataFrame(result).to_csv("out.csv")


if __name__ == '__main__':
    unittest.main()